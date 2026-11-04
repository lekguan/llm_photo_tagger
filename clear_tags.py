import os
import time
import json
import tempfile
import subprocess

PHOTO_DIR = "/mnt/synology_photos"
TAG_PREFIX = "st_"
CHUNK_SIZE = 100

def collect_all_image_paths(root_dir: str):
    print(f"[*] Discovering files in: {root_dir}...")
    t0 = time.time()
    valid_exts = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".xmp"}
    file_paths = []
    
    for root, _, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                file_paths.append(os.path.join(root, f))
                
    print(f"[+] Found {len(file_paths)} files in {time.time() - t0:.2f}s.\n")
    return file_paths

def check_chunk_for_tags(chunk: list):
    """Checks files in ONE batch ExifTool call over an argfile."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        f.write("\n".join(chunk) + "\n")
        temp_list = f.name

    cmd = [
        "exiftool", "-fast2", "-json", "-q",
        "-XMP:Subject", "-IPTC:Keywords", "-EXIF:XPKeywords",
        "-@", temp_list
    ]

    dirty_files = {}

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            for item in data:
                fpath = item.get("SourceFile")
                if not fpath:
                    continue
                
                tags_found = set()
                for key in ("Subject", "Keywords", "XPKeywords"):
                    val = item.get(key)
                    if isinstance(val, list):
                        for v in val:
                            if str(v).lower().startswith(TAG_PREFIX.lower()):
                                tags_found.add(str(v))
                    elif isinstance(val, str):
                        parts = [p.strip() for p in val.replace(";", ",").split(",") if p.strip()]
                        for p_val in parts:
                            if p_val.lower().startswith(TAG_PREFIX.lower()):
                                tags_found.add(p_val)
                                
                if tags_found:
                    dirty_files[fpath] = list(tags_found)
    except Exception as e:
        print(f"[!] Error checking chunk: {e}")
    finally:
        if os.path.exists(temp_list):
            os.remove(temp_list)

    return dirty_files

def strip_tags_bulk(dirty_files: dict):
    """Safely executes batch tag stripping via a verified args file."""
    if not dirty_files:
        return 0

    arg_lines = []
    
    for fpath, tags in dirty_files.items():
        # Clean any stale leftover tmp files before writing
        stale_tmp = f"{fpath}_exiftool_tmp"
        if os.path.exists(stale_tmp):
            try:
                os.remove(stale_tmp)
            except OSError:
                pass

        arg_lines.append("-overwrite_original")
        for tag in tags:
            arg_lines.append(f"-XMP:Subject-={tag}")
            arg_lines.append(f"-IPTC:Keywords-={tag}")
            arg_lines.append(f"-EXIF:XPKeywords-={tag}")
        arg_lines.append(fpath)
        arg_lines.append("-execute")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        f.write("\n".join(arg_lines) + "\n")
        temp_args = f.name

    cmd = ["exiftool", "-q", "-@", temp_args]

    success_count = 0
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if res.returncode == 0:
            success_count = len(dirty_files)
        else:
            print(f"[!] Exiftool write error: {res.stderr[:200]}")
    except Exception as e:
        print(f"[!] Write exception: {e}")
    finally:
        if os.path.exists(temp_args):
            os.remove(temp_args)

    return success_count

def main():
    all_files = collect_all_image_paths(PHOTO_DIR)
    total = len(all_files)
    if total == 0:
        print("[!] No images found. Exiting.")
        return

    print(f"[*] Starting robust chunked tag stripper (Chunk size: {CHUNK_SIZE})...\n")

    cleaned_total = 0
    scanned_total = 0
    clean_skipped = 0
    t_start = time.time()

    for i in range(0, total, CHUNK_SIZE):
        chunk = all_files[i : i + CHUNK_SIZE]
        
        # 1. Bulk check
        dirty_files = check_chunk_for_tags(chunk)
        
        # 2. Bulk strip only if dirty files found
        if dirty_files:
            stripped = strip_tags_bulk(dirty_files)
            cleaned_total += stripped
            
            sample_file = os.path.basename(list(dirty_files.keys())[0])
            extra = f" (+{len(dirty_files)-1} more)" if len(dirty_files) > 1 else ""
            
            if stripped > 0:
                print(f"[{min(i + len(chunk), total)}/{total}] [STRIPPED {stripped} FILES] {sample_file}{extra}")
            else:
                print(f"[{min(i + len(chunk), total)}/{total}] [WRITE FAILED for {len(dirty_files)} FILES] {sample_file}")
                
            clean_skipped += (len(chunk) - len(dirty_files))
        else:
            clean_skipped += len(chunk)

        scanned_total += len(chunk)

    print("\n" + "=" * 60)
    print(f"[COMPLETE] Tag Removal Finished in {time.time() - t_start:.2f}s:")
    print(f"  Total Files Checked : {scanned_total}")
    print(f"  Clean Files Skipped : {clean_skipped}")
    print(f"  Files Stripped      : {cleaned_total}")
    print("=" * 60)

if __name__ == "__main__":
    main()
