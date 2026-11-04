import os
import time
import json
import base64
import subprocess
import threading
from io import BytesIO
from queue import Queue
from PIL import Image
import requests

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PHOTO_DIR = "/mnt/synology_photos"
SERVER_URL = "http://127.0.0.1:9931/v1/chat/completions"
TAG_PREFIX = "st_"
MAX_WORKERS = 4
MAX_IMAGE_DIMENSION = 1024
CHUNK_SIZE = 250  # Number of files checked per exiftool batch call

# ----------------------------------------------------------------------
# Shared Memory Queues & Locks
# ----------------------------------------------------------------------
# Holds (filepath, base64_image) ready for immediate GPU inference
image_payload_queue = Queue(maxsize=30)

# Holds (filepath, tags, worker_id) awaiting background NAS write
write_queue = Queue(maxsize=100)

print_lock = threading.Lock()
stats = {"scanned": 0, "skipped": 0, "tagged": 0, "errors": 0}

# ----------------------------------------------------------------------
# Fast File Listing (Pure Directory Metadata, Zero Network Content Read)
# ----------------------------------------------------------------------
def collect_all_image_paths(root_dir: str):
    print(f"[*] Discovering image files in: {root_dir}...")
    t0 = time.time()
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    file_paths = []
    
    for root, _, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                file_paths.append(os.path.join(root, f))
                
    print(f"[+] Found {len(file_paths)} files in {time.time() - t0:.2f}s.")
    return file_paths

# ----------------------------------------------------------------------
# Chunked Exiftool Scanner (Feeds GPU Continuously)
# ----------------------------------------------------------------------
def scan_and_prefetch_producer(all_files: list):
    """
    Chunks all files into batches of CHUNK_SIZE, queries Exiftool via stdin list,
    and immediately feeds untagged photos to the GPU prefetch queue.
    """
    total = len(all_files)
    
    for i in range(0, total, CHUNK_SIZE):
        chunk = all_files[i : i + CHUNK_SIZE]
        
        # Prepare list of file paths separated by newlines for exiftool stdin
        input_file_list = "\n".join(chunk).encode("utf-8")
        
        cmd = [
            "exiftool", "-fast2", "-json", "-q",
            "-XMP:Subject", "-IPTC:Keywords", "-EXIF:XPKeywords",
            "-@", "-"  # Read file list from stdin
        ]
        
        untagged_batch = []
        skipped_count = 0
        
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate(input=input_file_list, timeout=45)
            
            if p.returncode == 0 and stdout.strip():
                data = json.loads(stdout.decode("utf-8"))
                for item in data:
                    fpath = item.get("SourceFile")
                    if not fpath:
                        continue
                    
                    all_tags = []
                    for key in ("Subject", "Keywords", "XPKeywords"):
                        val = item.get(key)
                        if isinstance(val, list):
                            all_tags.extend(val)
                        elif isinstance(val, str):
                            all_tags.extend([v.strip() for v in val.replace(";", ",").split(",") if v.strip()])

                    has_st = any(t.lower().startswith(TAG_PREFIX.lower()) for t in all_tags)
                    if has_st:
                        skipped_count += 1
                    else:
                        untagged_batch.append(fpath)
            else:
                # If chunk check fails, conservatively treat chunk as untagged
                untagged_batch = chunk
        except Exception as e:
            print(f"[!] Exiftool chunk scan error: {e}")
            untagged_batch = chunk

        with print_lock:
            stats["scanned"] += len(chunk)
            stats["skipped"] += skipped_count
            print(f"[{stats['scanned']}/{total}] Indexed chunk | Untagged found: {len(untagged_batch)} | Skipped: {skipped_count}")

        # Prefetch and resize untagged photos from this chunk directly into RAM
        for filepath in untagged_batch:
            try:
                with open(filepath, "rb") as f:
                    raw_bytes = f.read()
                with Image.open(BytesIO(raw_bytes)) as img:
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                # Push into payload queue for GPU workers
                image_payload_queue.put((filepath, b64_str))
            except Exception as e:
                print(f"[!] Read/Resize failed on {os.path.basename(filepath)}: {e}")

    # Send termination sentinel tokens to inference workers
    for _ in range(MAX_WORKERS):
        image_payload_queue.put(None)

# ----------------------------------------------------------------------
# VLM Inference (Direct HTTP, Zero Artificial Delays)
# ----------------------------------------------------------------------
def get_vlm_tags(b64_img: str):
    payload = {
        "model": "qwen3-vl-8b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this photo. Output ONLY a comma-separated or line-separated "
                            "list of 5 to 10 accurate, lowercase search keywords (e.g., baby clothes, "
                            "beach sunset, golden retriever). Do not write full sentences or explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    },
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=45)
        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"].strip().lower()
            lines = raw.replace("\n", ",").split(",")
            cleaned = []
            for item in lines:
                tag = item.strip().lstrip("-*• ").strip()
                if len(tag) > 1 and not tag.isdigit():
                    cleaned.append(tag)
            unique = list(dict.fromkeys(cleaned))[:10]
            return [f"{TAG_PREFIX}{t}" for t in unique]
    except Exception as e:
        print(f"[!] VLM request error: {e}")
    return None

# ----------------------------------------------------------------------
# Decoupled Background Exif Writer
# ----------------------------------------------------------------------
def background_writer_func():
    """Writes metadata asynchronously in the background."""
    while True:
        item = write_queue.get()
        if item is None:
            write_queue.task_done()
            break
            
        filepath, tags, worker_id = item
        filename = os.path.basename(filepath)

        cmd = ["exiftool", "-overwrite_original"]
        for tag in tags:
            cmd.append(f"-XMP:Subject+={tag}")
            cmd.append(f"-IPTC:Keywords+={tag}")
            cmd.append(f"-EXIF:XPKeywords+={tag}")
        cmd.append(filepath)

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
            success = (res.returncode == 0)
        except Exception:
            success = False

        with print_lock:
            if success:
                stats["tagged"] += 1
                print(f"[TAGGED #{stats['tagged']}] (W{worker_id}) {filename} -> {tags}")
            else:
                stats["errors"] += 1
                print(f"[WRITE_FAIL] (W{worker_id}) {filename}")

        write_queue.task_done()

# ----------------------------------------------------------------------
# Worker Loop
# ----------------------------------------------------------------------
def worker_thread_func(worker_id: int):
    while True:
        item = image_payload_queue.get()
        if item is None:
            image_payload_queue.task_done()
            break

        filepath, b64_img = item
        tags = get_vlm_tags(b64_img)

        if tags:
            write_queue.put((filepath, tags, worker_id))
        else:
            with print_lock:
                stats["errors"] += 1
                print(f"[!] [TAG_FAIL] (W{worker_id}) {os.path.basename(filepath)}")

        image_payload_queue.task_done()

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main():
    all_files = collect_all_image_paths(PHOTO_DIR)
    total = len(all_files)
    if total == 0:
        print("[!] No images found matching valid extensions. Exiting.")
        return

    print(f"[*] Starting streaming pipeline (Chunk size: {CHUNK_SIZE}, Workers: {MAX_WORKERS})...\n")

    # 1. Start background disk-writer thread
    writer = threading.Thread(target=background_writer_func, daemon=True)
    writer.start()

    # 2. Start GPU inference workers
    workers = [
        threading.Thread(target=worker_thread_func, args=(i,), daemon=True)
        for i in range(MAX_WORKERS)
    ]
    for w in workers:
        w.start()

    # 3. Start chunked producer in separate thread
    producer = threading.Thread(target=scan_and_prefetch_producer, args=(all_files,), daemon=True)
    producer.start()

    # 4. Wait for producer and GPU inference workers to complete
    producer.join()
    for w in workers:
        w.join()

    # 5. Flush background writer
    write_queue.put(None)
    writer.join()

    print("\n" + "=" * 60)
    print(f"[COMPLETE] Final Summary:")
    print(f"  Total Files Scanned : {stats['scanned']}")
    print(f"  Already Tagged      : {stats['skipped']}")
    print(f"  Newly Tagged by VLM : {stats['tagged']}")
    print(f"  Failures/Errors     : {stats['errors']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
