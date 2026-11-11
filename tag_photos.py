import base64
import json
import threading
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
)
from queue import Queue, Empty, Full

import requests

# ======================================================================
# Configuration
# ======================================================================
NAS_HOST = "http://192.168.1.100:5000"  # Synology DSM URL
DSM_USERNAME = "your_dsm_username"
DSM_PASSWORD = "your_dsm_password"
# ======================================================================
# Configuration
# ======================================================================
USE_SHARED_SPACE = False

SERVER_URL = "http://127.0.0.1:9931/v1/chat/completions"
VLM_MODEL = "qwen3-vl-8b"

# Maximum eligible photos per run. None means unlimited.
MAX_CANDIDATES = None

# "skip"      -> Add generated tags to photos without prefix tags.
# "overwrite" -> Replace prefix tags, preserving other tags.
# "clear"     -> Remove prefix tags only; do not run the VLM.
MODE = "skip"

TAG_PREFIX = "pt_"

# Path within the selected Photos space, NOT /volume1/... or an SMB path.
# Replace this placeholder before running. None explicitly allows the whole space.
TARGET_DIRECTORY = "/MobileBackup"
INCLUDE_SUBDIRECTORIES = True

BATCH_FETCH_SIZE = 500
VERIFY_TLS = True

API_TIMEOUT = (10, 120)
VLM_TIMEOUT = 60

VERIFY_ATTEMPTS = 5
VERIFY_DELAY = 2

# Change INFERENCE_WORKERS to 1 to compare against sequential inference.
INFERENCE_WORKERS = 4

# Used in skip mode. Overwrite and clear use one writer.
NAS_WRITERS = 2

# Bounds the number of completed preparations waiting for writers.
PREPARE_QUEUE_SIZE = 8


# ======================================================================
# Session and shared state
# ======================================================================
ENTRY_URL = f"{NAS_HOST.rstrip('/')}/webapi/entry.cgi"

API_PREFIX = "SYNO.FotoTeam" if USE_SHARED_SPACE else "SYNO.Foto"
API_ITEM = f"{API_PREFIX}.Browse.Item"
API_GENERAL_TAG = f"{API_PREFIX}.Browse.GeneralTag"
API_THUMB = f"{API_PREFIX}.Thumbnail"
API_FOLDER = f"{API_PREFIX}.Browse.Folder"
allowed_folder_ids = None
folder_paths = {}

session = requests.Session()
session.verify = VERIFY_TLS

tag_catalog_cache = None
catalog_lock = threading.RLock()

api_thread_local = threading.local()
writer_sessions = []
writer_sessions_lock = threading.Lock()


def clone_nas_session():
    client = requests.Session()

    client.verify = session.verify
    client.headers.update(session.headers)
    client.cookies.update(session.cookies)
    client.proxies.update(session.proxies)
    client.auth = session.auth
    client.cert = session.cert
    client.trust_env = session.trust_env

    return client


def initialize_writer_session():
    client = clone_nas_session()
    api_thread_local.session = client

    with writer_sessions_lock:
        writer_sessions.append(client)


def current_api_session():
    return getattr(api_thread_local, "session", session)


# ======================================================================
# Checked API helpers
# ======================================================================
def api_call(api, method, version=1, **params):
    payload = {
        "api": api,
        "version": str(version),
        "method": method,
        **params,
    }

    response = current_api_session().post(
        ENTRY_URL,
        data=payload,
        timeout=API_TIMEOUT,
    )
    response.raise_for_status()

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"{api}.{method}: non-JSON response: "
            f"{response.text[:500]}"
        ) from exc

    if not isinstance(result, dict) or result.get("success") is not True:
        raise RuntimeError(
            f"{api}.{method} rejected: "
            f"{json.dumps(result, ensure_ascii=False)}"
        )

    data = result.get("data", {})

    if isinstance(data, dict) and data.get("error_list"):
        raise RuntimeError(
            f"{api}.{method}: per-item errors: "
            f"{json.dumps(data['error_list'], ensure_ascii=False)}"
        )

    return data


def require_list(data, context):
    if not isinstance(data, dict) or not isinstance(data.get("list"), list):
        raise RuntimeError(
            f"{context}: expected data.list, received: "
            f"{json.dumps(data, ensure_ascii=False)[:1000]}"
        )

    return data["list"]


def item_tags(item):
    additional = item.get("additional")

    if not isinstance(additional, dict) or "tag" not in additional:
        raise RuntimeError(
            f"Item {item.get('id')}: tag metadata is missing; "
            "refusing to assume the photo has no tags."
        )

    tags = additional["tag"]

    if not isinstance(tags, list):
        raise RuntimeError(
            f"Item {item.get('id')}: unexpected tag structure."
        )

    for tag in tags:
        if not isinstance(tag, dict) or not isinstance(tag.get("name"), str):
            raise RuntimeError(
                f"Unexpected tag object: "
                f"{json.dumps(tag, ensure_ascii=False)}"
            )

    return tags


def is_generated(name):
    return name.lower().startswith(TAG_PREFIX.lower())


def numeric_tag_ids(tags):
    ids = []

    for tag in tags:
        value = tag.get("id")

        if value is None or isinstance(value, bool):
            raise RuntimeError(
                f"Missing or invalid tag ID: "
                f"{json.dumps(tag, ensure_ascii=False)}"
            )

        try:
            ids.append(int(value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid numeric tag ID: {value!r}"
            ) from exc

    return list(dict.fromkeys(ids))


# ======================================================================
# Authentication and scanning
# ======================================================================
def login():
    api_call(
        "SYNO.API.Auth",
        "login",
        version=3,
        account=DSM_USERNAME,
        passwd=DSM_PASSWORD,
        format="cookie",
    )

    print("[+] Logged into Synology Photos.")


def normalize_photo_directory(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("TARGET_DIRECTORY must be a non-empty Photos path or None.")
    value = value.strip()
    if "\\" in value:
        raise ValueError("Use a Photos-relative path with forward slashes.")
    parts = [p for p in value.split("/") if p]
    if any(p in {".", ".."} for p in parts):
        raise ValueError("Directory paths must not contain '.' or '..'.")
    return "/" + "/".join(parts)


def scan_read_call(api, **params):
    for attempt in range(1, 4):
        try:
            return api_call(api, "list", **params)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == 3:
                raise
            print(f"[SCAN RETRY] {api}.list | {attempt}/3", flush=True)
            time.sleep(5 * attempt)


def list_child_photo_folders(parent_id=None):
    result, seen = [], set()
    offset = 0
    while True:
        params = {"offset": offset, "limit": BATCH_FETCH_SIZE}
        if parent_id is not None:
            params["id"] = int(parent_id)
        page = require_list(scan_read_call(API_FOLDER, **params), "Folder.list")
        for folder in page:
            if (not isinstance(folder, dict) or not isinstance(folder.get("name"), str)
                    or folder.get("id") is None or isinstance(folder.get("id"), bool)):
                raise RuntimeError(f"Unexpected folder object: {folder!r}")
            fid = int(folder["id"])
            if fid in seen:
                raise RuntimeError("Folder pagination repeated an ID.")
            seen.add(fid)
            result.append(folder)
        offset += len(page)
        if len(page) < BATCH_FETCH_SIZE:
            return result


def child_folder_path(parent_path, reported_name):
    path = normalize_photo_directory(reported_name)
    if parent_path == "/":
        return path
    if path.startswith(parent_path + "/"):
        return path
    if "/" not in path.strip("/"):
        return parent_path + path
    raise RuntimeError(f"Unexpected child path {path!r} beneath {parent_path!r}.")


def configure_directory_scope():
    global allowed_folder_ids, folder_paths
    allowed_folder_ids = None
    folder_paths = {0: "/"}
    if not isinstance(INCLUDE_SUBDIRECTORIES, bool):
        raise ValueError("INCLUDE_SUBDIRECTORIES must be True or False.")
    target = normalize_photo_directory(TARGET_DIRECTORY) if TARGET_DIRECTORY is not None else "/"
    if target == "/" and TARGET_DIRECTORY is not None and not INCLUDE_SUBDIRECTORIES:
        raise ValueError("Root-only mode is unsupported; choose a named folder.")
    if target == "/":
        print("[*] Reading folder paths for the selected Photos space...", flush=True)
        pending = [(None, "/")]
        visited = set()
    else:
        parent_id, resolved = None, ""
        print(f"[*] Resolving Photos directory: {target}", flush=True)
        for component in target.strip("/").split("/"):
            expected = f"{resolved}/{component}"
            children = list_child_photo_folders(parent_id)
            matches = [f for f in children
                       if normalize_photo_directory(f["name"]) in {expected, f"/{component}"}]
            if len(matches) != 1:
                available = "\n".join(f"  {f['name']} (ID:{f['id']})" for f in children[:40])
                raise RuntimeError(f"Could not uniquely resolve {expected!r}. No photos will be processed."
                                   f"\nAvailable children (up to 40):\n{available}")
            parent_id, resolved = int(matches[0]["id"]), expected
            folder_paths[parent_id] = resolved
            print(f"    Resolved: {resolved} (ID:{parent_id})", flush=True)
        allowed_folder_ids = {parent_id}
        pending = [(parent_id, target)] if INCLUDE_SUBDIRECTORIES else []
        visited = {parent_id}
    while pending:
        parent_id, parent_path = pending.pop()
        for folder in list_child_photo_folders(parent_id):
            fid = int(folder["id"])
            if fid in visited:
                raise RuntimeError("Folder traversal repeated an ID; refusing ambiguous scope.")
            visited.add(fid)
            path = child_folder_path(parent_path, folder["name"])
            folder_paths[fid] = path
            if allowed_folder_ids is not None:
                allowed_folder_ids.add(fid)
            pending.append((fid, path))
    print(f"[+] Directory scope: {target} | "
          f"Folders: {len(allowed_folder_ids) if allowed_folder_ids is not None else len(visited)}", flush=True)


def photo_is_in_scope(item):
    if allowed_folder_ids is None:
        return True
    value = item.get("folder_id")
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"Photo {item.get('id')}: missing/invalid folder_id.")
    return int(value) in allowed_folder_ids


def get_candidates():
    print(f"[*] Scanning selected scope | Mode: {MODE.upper()} "
          f"| Candidate limit: {MAX_CANDIDATES}", flush=True)
    candidates, seen = [], set()
    scanned = skipped = 0
    started = time.perf_counter()
    fields = ["tag"] if MODE == "clear" else ["tag", "thumbnail"]
    folder_ids = sorted(allowed_folder_ids) if allowed_folder_ids is not None else [None]
    def finish(reason):
        elapsed = time.perf_counter() - started
        print(f"[+] Scan stopped: {reason}. Checked: {scanned} | Queued: {len(candidates)} "
              f"| Skipped: {skipped} | Time: {elapsed:.2f}s\n", flush=True)
        return candidates
    for folder_id in folder_ids:
        offset, folder_seen = 0, set()
        if folder_id is not None:
            print(f"    Browsing: {folder_paths[folder_id]} (ID:{folder_id})", flush=True)
        while True:
            limit = BATCH_FETCH_SIZE
            if MAX_CANDIDATES is not None:
                limit = min(limit, MAX_CANDIDATES - len(candidates))
            params = {"offset": offset, "limit": limit, "additional": json.dumps(fields)}
            if folder_id is not None:
                params["folder_id"] = int(folder_id)
            items = require_list(scan_read_call(API_ITEM, **params), "Item.list")
            if not items:
                break
            for item in items:
                scanned += 1
                item_id = int(item["id"])
                if item_id in folder_seen:
                    raise RuntimeError("Photo pagination repeated an ID.")
                folder_seen.add(item_id)
                actual = item.get("folder_id")
                if actual is None or isinstance(actual, bool):
                    raise RuntimeError(f"Photo {item_id} has no valid folder_id.")
                actual = int(actual)
                if folder_id is not None and actual != folder_id:
                    raise RuntimeError(f"Listing for folder {folder_id} returned photo {item_id} "
                                       f"with folder_id={actual}. No writes have started.")
                if item_id in seen:
                    skipped += 1
                    continue
                seen.add(item_id)
                if not photo_is_in_scope(item):
                    raise RuntimeError("Photo outside configured scope.")
                tags = item_tags(item)
                generated = any(is_generated(t["name"]) for t in tags)
                if not ((MODE == "skip" and not generated) or (MODE == "clear" and generated)
                        or MODE == "overwrite"):
                    skipped += 1
                    continue
                if actual not in folder_paths:
                    raise RuntimeError(f"Unknown folder ID {actual}; rerun to refresh folder paths.")
                filename = item.get("filename", f"photo_{item_id}")
                full_path = folder_paths[actual].rstrip("/") + "/" + filename
                thumb = item["additional"].get("thumbnail") or {}
                candidates.append({"id": item_id, "filename": filename, "full_path": full_path,
                                   "cache_key": thumb.get("cache_key") or item.get("cache_key"),
                                   "thumbnail_unit_id": thumb.get("unit_id")})
                if MAX_CANDIDATES is not None and len(candidates) >= MAX_CANDIDATES:
                    return finish("candidate limit reached")
            offset += len(items)
            if len(items) < limit:
                break
    return finish("end of selected scope")


def read_photo_tags(item_id):
    data = api_call(
        API_ITEM,
        "get",
        version=2,
        id=json.dumps([int(item_id)]),
        additional=json.dumps(["tag"]),
    )

    items = require_list(data, "Item.get")

    matching = [
        item for item in items
        if int(item.get("id", -1)) == int(item_id)
    ]

    if len(matching) != 1:
        raise RuntimeError(
            f"Could not uniquely read photo {item_id}."
        )

    item = matching[0]
    if not photo_is_in_scope(item):
        raise RuntimeError(f"Photo {item_id} is outside configured scope; refusing processing.")
    return item_tags(item)


# ======================================================================
# Thumbnail and local VLM
# ======================================================================
def fetch_thumb(
    item_id,
    cache_key=None,
    thumbnail_unit_id=None,
    http_session=None,
):
    client = http_session if http_session is not None else session

    if thumbnail_unit_id is None:
        raise RuntimeError(
            f"Photo {item_id}: thumbnail unit_id is missing."
        )

    params = {
        "api": API_THUMB,
        "version": "2",
        "method": "get",
        "id": int(thumbnail_unit_id),
        "type": "unit",
        "size": "xl",
    }

    if cache_key:
        params["cache_key"] = cache_key

    response = client.get(
        ENTRY_URL,
        params=params,
        timeout=API_TIMEOUT,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if not content_type.startswith("image/"):
        raise RuntimeError(
            "Thumbnail request did not return an image: "
            f"{response.text[:500]}"
        )

    mime_type = content_type.split(";", 1)[0]
    encoded = base64.b64encode(response.content).decode("ascii")

    return mime_type, encoded


def get_vlm_tags(mime_type, encoded_image):
    prompt = (
        "What natural search queries would someone type into a photo "
        "search bar months later to find this exact picture?\n\n"
        "Rules:\n"
        "1. Focus on specific nouns: subject, document type, setting, "
        "event, activity, or food (e.g., boarding pass, receipt, wedding, "
        "birthday cake, golden retriever, beach sunset, hiking).\n"
        "2. Include common everyday categories (e.g., both "
        "'golden retriever' and 'dog', 'receipt' and 'expense').\n"
        "3. Do NOT describe visual aesthetics or photography styles "
        "(never use 'bokeh', 'close-up', 'warm lighting', 'indoor shot', "
        "'blurry background', 'view from above').\n"
        "4. Output 5 to 10 comma-separated, lowercase keywords or "
        "short phrases. No sentences.\n"
        "5. Search matches substrings within tags. Prefer specific phrases that "
        "also contain useful broader search terms. Avoid redundant tags when "
        "one tag already contains another (e.g., use 'golden retriever' without "
        "also adding 'retriever'). Keep broader category tags when their words "
        "are not already present (e.g., also include 'dog')."
    )

    response = requests.post(
        SERVER_URL,
        json={
            "model": VLM_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,{encoded_image}"
                            )
                        },
                    },
                ],
            }],
            "temperature": 0.2,
            "max_tokens": 64,
        },
        timeout=VLM_TIMEOUT,
    )
    response.raise_for_status()

    result = response.json()

    try:
        raw = result["choices"][0]["message"]["content"].strip().lower()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            "VLM returned an unexpected response structure."
        ) from exc

    tags = []

    for part in raw.replace("\n", ",").split(","):
        name = part.strip().lstrip("-*• ").strip("\"' ")

        if len(name) <= 1 or name.isdigit():
            continue

        if not name.startswith(TAG_PREFIX.lower()):
            name = f"{TAG_PREFIX}{name}"

        if name not in tags:
            tags.append(name)

    if not tags:
        raise RuntimeError(f"VLM returned no usable tags: {raw!r}")

    return tags[:10]


# ======================================================================
# Tag catalog
# ======================================================================
def extract_created_tag_id(data, expected_name):
    if not isinstance(data, dict):
        return None

    candidates = [data]

    if isinstance(data.get("tag"), dict):
        candidates.append(data["tag"])

    if isinstance(data.get("list"), list):
        candidates.extend(data["list"])

    for candidate in candidates:
        if not isinstance(candidate, dict) or "id" not in candidate:
            continue

        name = candidate.get("name")

        if name is not None and name != expected_name:
            continue

        value = candidate["id"]

        if isinstance(value, bool):
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def resolve_tag_ids(names, force_refresh=False):
    """Create/resolve uncached names without downloading the tag catalog.

    Requires the NAS to return an existing tag ID for an existing name.
    """
    global tag_catalog_cache
    with catalog_lock:
        if force_refresh or tag_catalog_cache is None:
            tag_catalog_cache = {}
        ids = []
        for name in dict.fromkeys(names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Invalid tag name: {name!r}")
            if name in tag_catalog_cache:
                ids.append(tag_catalog_cache[name])
                continue
            try:
                data = api_call(API_GENERAL_TAG, "create", name=json.dumps(name, ensure_ascii=False))
            except Exception as exc:
                raise RuntimeError(
                    f"GeneralTag.create failed for {name!r}: {exc}. "
                    "No catalog download or automatic create retry was performed. "
                    "The server may have processed the request despite a lost response."
                ) from exc
            tag_id = extract_created_tag_id(data, expected_name=name)
            if tag_id is None:
                raise RuntimeError(
                    f"GeneralTag.create succeeded for {name!r}, but returned no usable ID. "
                    "No catalog download was performed. Response data: "
                    f"{json.dumps(data, ensure_ascii=False)[:2000]}"
                )
            tag_catalog_cache[name] = tag_id
            ids.append(tag_id)
        return list(dict.fromkeys(ids))


def add_tag_ids(item_id, tag_ids):
    global tag_catalog_cache

    if not tag_ids:
        return

    try:
        api_call(
            API_ITEM,
            "add_tag",
            id=json.dumps([int(item_id)]),
            tag=json.dumps(tag_ids),
        )

    except Exception:
        with catalog_lock:
            tag_catalog_cache = None

        raise


def remove_tag_ids(item_id, tag_ids):
    global tag_catalog_cache

    if not tag_ids:
        return

    try:
        api_call(
            API_ITEM,
            "remove_tag",
            id=json.dumps([int(item_id)]),
            tag=json.dumps(tag_ids),
        )

    finally:
        with catalog_lock:
            tag_catalog_cache = None


def wait_for_tags(item_id, required_names, forbidden_names):
    required_names = set(required_names)
    forbidden_names = set(forbidden_names)
    last_names = set()

    for attempt in range(VERIFY_ATTEMPTS):
        tags = read_photo_tags(item_id)
        last_names = {tag["name"] for tag in tags}

        missing = required_names - last_names
        remaining = forbidden_names & last_names

        if not missing and not remaining:
            return tags

        if attempt + 1 < VERIFY_ATTEMPTS:
            time.sleep(VERIFY_DELAY)

    raise RuntimeError(
        "Read-back verification failed. "
        f"Missing expected tags: {sorted(required_names - last_names)}; "
        f"obsolete tags still present: "
        f"{sorted(forbidden_names & last_names)}. "
        "No automatic write retry was performed."
    )


def apply_generated_tags(item_id, new_names):
    before = read_photo_tags(item_id)
    before_names = {tag["name"] for tag in before}

    old_generated = {
        name for name in before_names if is_generated(name)
    }
    preserved = before_names - old_generated
    new_names = set(new_names)

    if MODE == "skip" and old_generated:
        return "SKIPPED"

    desired_names = (
        before_names | new_names
        if MODE == "skip"
        else preserved | new_names
    )

    obsolete_names = (
        old_generated - new_names
        if MODE == "overwrite"
        else set()
    )

    names_to_add = sorted(new_names - before_names)

    if names_to_add:
        tag_ids = resolve_tag_ids(names_to_add)
        add_tag_ids(item_id, tag_ids)

    current = wait_for_tags(
        item_id,
        required_names=before_names | new_names,
        forbidden_names=set(),
    )

    # Skip mode needs only this one successful verification.
    if MODE == "skip":
        return "TAGGED"

    if obsolete_names:
        obsolete_tags = [
            tag for tag in current
            if tag["name"] in obsolete_names
        ]

        remove_tag_ids(item_id, numeric_tag_ids(obsolete_tags))

    final_tags = wait_for_tags(
        item_id,
        required_names=desired_names,
        forbidden_names=obsolete_names,
    )

    final_generated = {
        tag["name"] for tag in final_tags
        if is_generated(tag["name"])
    }

    if final_generated != new_names:
        raise RuntimeError(
            "Overwrite verification found an unexpected generated-tag "
            f"set: {sorted(final_generated)}"
        )

    return "OVERWROTE"


def clear_tags(item_id):
    before = read_photo_tags(item_id)

    targets = [
        tag for tag in before if is_generated(tag["name"])
    ]

    if not targets:
        return "SKIPPED", 0

    target_names = {tag["name"] for tag in targets}

    preserved_names = {
        tag["name"] for tag in before
        if not is_generated(tag["name"])
    }

    remove_tag_ids(item_id, numeric_tag_ids(targets))

    after = wait_for_tags(
        item_id,
        required_names=preserved_names,
        forbidden_names=target_names,
    )

    remaining_generated = {
        tag["name"] for tag in after
        if is_generated(tag["name"])
    }

    if remaining_generated:
        raise RuntimeError(
            "Clear verification found matching tags still present: "
            f"{sorted(remaining_generated)}"
        )

    return "CLEARED", len(targets)


# ======================================================================
# Preparation pipeline
# ======================================================================
def queue_put_until_stopped(output_queue, value, stop_event):
    while not stop_event.is_set():
        try:
            output_queue.put(value, timeout=0.25)
            return True
        except Full:
            continue

    return False


def prepare_photos(candidates, output_queue, stop_event):
    """One downloader feeds inference workers that publish results directly."""
    thumbnail_queue = Queue(maxsize=INFERENCE_WORKERS)
    workers = []
    def inference_worker():
        while not stop_event.is_set():
            try:
                job = thumbnail_queue.get(timeout=0.25)
            except Empty:
                continue
            try:
                if job is None:
                    return
                index, item, mime, encoded = job
                prepared = {"index": index, "item": item, "names": None, "error": None}
                try:
                    prepared["names"] = get_vlm_tags(mime, encoded)
                    if not prepared["names"]:
                        raise RuntimeError("VLM returned no usable tags.")
                except Exception as exc:
                    prepared["error"] = str(exc)
                if not queue_put_until_stopped(output_queue, prepared, stop_event):
                    return
                del job, encoded, prepared
            finally:
                thumbnail_queue.task_done()
    try:
        for number in range(INFERENCE_WORKERS):
            worker = threading.Thread(target=inference_worker, name=f"vlm-{number + 1}", daemon=True)
            worker.start()
            workers.append(worker)
        with clone_nas_session() as client:
            for index, item in enumerate(candidates, 1):
                if stop_event.is_set():
                    break
                try:
                    mime, encoded = fetch_thumb(item["id"], item.get("cache_key"),
                                               item.get("thumbnail_unit_id"), http_session=client)
                except Exception as exc:
                    if not queue_put_until_stopped(output_queue,
                        {"index": index, "item": item, "names": None, "error": str(exc)}, stop_event):
                        break
                    continue
                if not queue_put_until_stopped(thumbnail_queue, (index, item, mime, encoded), stop_event):
                    break
                del encoded
    except Exception as exc:
        queue_put_until_stopped(output_queue, {"pipeline_error": str(exc)}, stop_event)
    finally:
        if not stop_event.is_set():
            for _ in workers:
                if not queue_put_until_stopped(thumbnail_queue, None, stop_event):
                    break
        for worker in workers:
            worker.join()
        queue_put_until_stopped(output_queue, None, stop_event)


# ======================================================================
# Main
# ======================================================================
def require_positive_integer(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")


def main():
    global tag_catalog_cache

    if MODE not in {"skip", "overwrite", "clear"}:
        raise ValueError(
            "MODE must be 'skip', 'overwrite', or 'clear'."
        )

    if not isinstance(TAG_PREFIX, str) or not TAG_PREFIX.strip():
        raise ValueError("TAG_PREFIX must be non-empty.")

    for name, value in (
        ("BATCH_FETCH_SIZE", BATCH_FETCH_SIZE),
        ("VERIFY_ATTEMPTS", VERIFY_ATTEMPTS),
        ("PREPARE_QUEUE_SIZE", PREPARE_QUEUE_SIZE),
        ("INFERENCE_WORKERS", INFERENCE_WORKERS),
        ("NAS_WRITERS", NAS_WRITERS),
    ):
        require_positive_integer(name, value)

    if MAX_CANDIDATES is not None:
        require_positive_integer("MAX_CANDIDATES", MAX_CANDIDATES)

    with catalog_lock:
        tag_catalog_cache = None

    login()
    configure_directory_scope()
    candidates = get_candidates()

    if not candidates:
        print("[*] No matching photos. Nothing to do.")
        return

    space = (
        "Shared Space"
        if USE_SHARED_SPACE
        else "Personal Space"
    )

    writer_count = NAS_WRITERS if MODE == "skip" else 1

    if MODE in {"overwrite", "clear"}:
        print(f"Mode             : {MODE.upper()}")
        print(f"Space            : {space}")
        print(f"Photos queued    : {len(candidates)}")
        print(f"Tag prefix       : {TAG_PREFIX!r}")
        print(f"Directory        : {TARGET_DIRECTORY if TARGET_DIRECTORY is not None else 'ALL'}")
        print(f"Subdirectories   : {INCLUDE_SUBDIRECTORIES}")
        print("NAS writers      : 1 (removal safety)")

        if MODE == "clear":
            print("Action           : Remove matching tags only.")
        else:
            print("Action           : Replace generated tags only.")

        print("Other tags       : Preserve and verify.")

        try:
            answer = input(
                "\nType 'yes' to continue: "
            ).strip().lower()

        except (KeyboardInterrupt, EOFError):
            print("\n[CANCELLED] No photos processed.")
            return

        if answer != "yes":
            print("[CANCELLED] No photos processed.")
            return

        print()

    started = time.perf_counter()

    succeeded = 0
    failed = 0
    skipped = 0

    interrupted = False
    pipeline_failed = False
    abnormal_exit = False

    stop_event = threading.Event()
    producer = None
    writer_pool = None
    pending_writes = {}

    def report_result(item, status, detail):
        completed = succeeded + failed + skipped
        elapsed = time.perf_counter() - started
        rate = succeeded / max(elapsed, 0.01)

        print(
            f"[{completed}/{len(candidates)}] [{status}] "
            f"{item['full_path']} (ID:{item['id']}) "
            f"| Average verified: {rate:.2f} photos/sec "
            f"-> {detail}",
            flush=True,
        )

    def collect_writes(block=False):
        nonlocal succeeded, failed, skipped

        if not pending_writes:
            return

        done, _ = wait(
            tuple(pending_writes),
            timeout=0.25 if block else 0,
            return_when=FIRST_COMPLETED,
        )

        for future in done:
            # Leave the future tracked until its result is accounted for.
            item, names = pending_writes[future]

            if future.cancelled():
                pending_writes.pop(future)
                continue

            try:
                status = future.result()

            except Exception as exc:
                failed += 1
                pending_writes.pop(future)
                report_result(item, "FAILED", str(exc))

            else:
                if status == "SKIPPED":
                    skipped += 1
                    detail = "generated tags already present"
                else:
                    succeeded += 1
                    detail = str(names)

                pending_writes.pop(future)
                report_result(item, status, detail)

    try:
        if MODE == "clear":
            for item in candidates:
                try:
                    status, removed_count = clear_tags(item["id"])

                except Exception as exc:
                    failed += 1
                    report_result(item, "FAILED", str(exc))

                else:
                    if status == "SKIPPED":
                        skipped += 1
                        detail = "no matching tags on fresh read"
                    else:
                        succeeded += 1
                        detail = f"removed {removed_count} tags"

                    report_result(item, status, detail)

        else:
            print(
                "[*] Pipeline active: "
                "1 thumbnail downloader "
                f"+ {INFERENCE_WORKERS} VLM workers "
                f"+ {writer_count} verified NAS writers.\n"
            )

            prepared_queue = Queue(maxsize=PREPARE_QUEUE_SIZE)

            writer_pool = ThreadPoolExecutor(
                max_workers=writer_count,
                initializer=initialize_writer_session,
                thread_name_prefix="nas-writer",
            )

            producer = threading.Thread(
                target=prepare_photos,
                args=(
                    candidates,
                    prepared_queue,
                    stop_event,
                ),
                daemon=True,
                name="photo-preparation",
            )
            producer.start()

            preparation_done = False

            while not preparation_done or pending_writes:
                collect_writes()

                # Bound submitted writer jobs to the actual writer count.
                if len(pending_writes) >= writer_count:
                    collect_writes(block=True)
                    continue

                if preparation_done:
                    if pending_writes:
                        collect_writes(block=True)
                    continue

                try:
                    prepared = prepared_queue.get(timeout=0.25)

                except Empty:
                    if producer.is_alive():
                        continue

                    # Check once more after observing thread termination.
                    try:
                        prepared = prepared_queue.get_nowait()

                    except Empty:
                        pipeline_failed = True
                        preparation_done = True

                        print(
                            "[PIPELINE FAILED] Preparation stopped "
                            "without its completion marker."
                        )
                        continue

                if prepared is None:
                    preparation_done = True
                    continue

                if "pipeline_error" in prepared:
                    pipeline_failed = True
                    print(
                        "[PIPELINE FAILED] "
                        f"{prepared['pipeline_error']}"
                    )
                    continue

                item = prepared["item"]

                if prepared["error"] is not None:
                    failed += 1
                    report_result(
                        item,
                        "PREPARE FAILED",
                        prepared["error"],
                    )
                    continue

                names = prepared["names"]

                future = writer_pool.submit(
                    apply_generated_tags,
                    item["id"],
                    names,
                )
                pending_writes[future] = (item, names)

    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()

        print(
            "\n[INTERRUPTED] Stopping new work. "
            "Waiting for running requests and NAS jobs to finish. "
            "Pressing Ctrl+C again may leave writes unverified."
        )

    except BaseException:
        abnormal_exit = True
        raise

    finally:
        stop_event.set()

        if writer_pool is not None:
            for future in pending_writes:
                future.cancel()

            writer_pool.shutdown(wait=True)

            while pending_writes:
                collect_writes(block=True)

        if producer is not None:
            # Preparation has no tag-write authority, but wait for
            # running VLM calls so no worker pool is left behind.
            producer.join()

        with writer_sessions_lock:
            for client in writer_sessions:
                client.close()

            writer_sessions.clear()

    elapsed = time.perf_counter() - started
    accounted_for = succeeded + failed + skipped
    unprocessed = len(candidates) - accounted_for

    if interrupted:
        heading = "STOPPED"
    elif pipeline_failed or abnormal_exit or unprocessed:
        heading = "PIPELINE INCOMPLETE"
    else:
        heading = "COMPLETE"

    print("\n" + "=" * 64)
    print(heading)
    print(f"  Mode                 : {MODE.upper()}")
    print(
        f"  Inference workers    : "
        f"{INFERENCE_WORKERS if MODE != 'clear' else 0}"
    )
    print(f"  NAS writers          : {writer_count}")
    print(f"  Photos queued        : {len(candidates)}")
    print(f"  Verified successful  : {succeeded}")
    print(f"  Skipped during run   : {skipped}")
    print(f"  Failed / unverified  : {failed}")
    print(f"  Unprocessed          : {unprocessed}")
    print(f"  Elapsed              : {elapsed:.2f}s")
    print(
        f"  Verified speed       : "
        f"{succeeded / max(elapsed, 0.01):.2f} photos/sec"
    )
    print("=" * 64)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(
            "\n[CANCELLED] If shutdown was interrupted, "
            "an in-flight write may be unverified."
        )
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        raise
