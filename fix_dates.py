import os
import re
import time
import json
import tempfile
import subprocess
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PHOTO_DIR = "/mnt/synology_photos"
CHUNK_SIZE = 100  # Conservative batch size to prevent SMB write timeouts

# Matches Synology structure: .../YYYY/MM/filename (e.g., .../2023/01/photo.jpg)
FOLDER_DATE_REGEX = re.compile(r".*[/\\](\d{4})[/\\](\d{2})[/\\][^/\\]+$")

def collect_all_image_paths(root_dir: str):
    """Fast OS directory walk (reads directory entries, zero file reads)."""
    print(f"[*] Discovering files in: {root_dir}...")
    t0 = time.time()
    valid_exts = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
    file_paths = []

    for root, _, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                file_paths.append(os.path.join(root, f))

    print(f"[+] Found {len(file_paths)} files in {time.time() - t0:.2f}s.\n")
    return file_paths

def check_chunk_for_missing_dates(chunk: list):
    """
    Checks chunk files via one ExifTool batch call.
    Returns: list of filepaths where Date Taken is NOT filled in.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        f.write("\n".join(chunk) + "\n")
        temp_list = f.name

    cmd = [
        "exiftool", "-fast2", "-json", "-q",
        "-DateTimeOriginal", "-CreateDate",
        "-@", temp_list
    ]

    missing_date_files = []

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            for item in data:
                fpath = item.get("SourceFile")
                if not fpath:
                    continue

                dto = item.get("DateTimeOriginal")
                cdate = item.get("CreateDate")

                # Check if Date Taken is already present
                has_date = bool(dto and str(dto).strip() and str(dto) != "0000:00:00 00:00:00")
                if not has_date:
                    has_date = bool(cdate and str(cdate).strip() and str(cdate) != "0000:00:00 00:00:00")

                if not has_date:
                    missing_date_files.append(fpath)
    except Exception as e:
        print(f"[!] Error checking chunk: {e}")
    finally:
        if os.path.exists(temp_list):
            os.remove(temp_list)

    return missing_date_files

def calculate_target_date(filepath: str):
    """
    Evaluates whether the path matches YYYY/MM/xyz and compares atime vs folder date.
    Returns: Formatted string 'YYYY:MM:DD HH:MM:SS' or None if folder doesn't match.
    """
    match = FOLDER_DATE_REGEX.match(filepath)
    if not match:
        return None

    folder_year = int(match.group(1))
    folder_month = int(match.group(2))

    # Validate valid calendar year/month
    if not (1900 <= folder_year <= 2100 and 1 <= folder_month <= 12):
        return None

    # Target folder date: 1st of that month at 12:00:00
    folder_dt = datetime(folder_year, folder_month, 1, 12, 0, 0)

    # Get last accessed time (atime) from file system
    try:
        file_stat = os.stat(filepath)
        atime_dt = datetime.fromtimestamp(file_stat.st_atime)
    except Exception:
        # Fallback to 1st of month if stat fails
        return folder_dt.strftime("%Y:%m:%d %H:%M:%S")

    # Calculate difference (6 months is ~182.5 days)
    diff_days = abs((atime_dt - folder_dt).total_seconds()) / 86400.0

    if diff_days > 182.5:
        # More than 6 months difference -> 1st of that month at 12:00:00
        chosen_dt = folder_dt
    else:
        # Within 6 months -> use last accessed timestamp
        chosen_dt = atime_dt

    return chosen_dt.strftime("%Y:%m:%d %H:%M:%S")

def apply_dates_bulk(updates_dict: dict):
    """
    Writes -AllDates atomically via ExifTool argfile and clears stale temp files.
    """
    if not updates_dict:
        return 0

    arg_lines = []

    for fpath, date_str in updates_dict.items():
        # Remove leftover temp file if a previous run was interrupted
        stale_tmp = f"{fpath}_exiftool_tmp"
        if os.path.exists(stale_tmp):
            try:
                os.remove(stale_tmp)
            except OSError:
                pass

        arg_lines.append("-overwrite_original")
        arg_lines.append(f"-AllDates={date_str}")
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
            success_count = len(updates_dict)
        else:
            print(f"[!] Exiftool date write error: {res.stderr[:200]}")
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

    print(f"[*] Starting chunked Date Taken repair (Chunk size: {CHUNK_SIZE})...\n")

    fixed_total = 0
    scanned_total = 0
    has_date_skipped = 0
    folder_mismatch_skipped = 0
    t_start = time.time()

    for i in range(0, total, CHUNK_SIZE):
        chunk = all_files[i : i + CHUNK_SIZE]

        # 1. Check which files are missing Date Taken
        missing_dates = check_chunk_for_missing_dates(chunk)
        has_date_skipped += (len(chunk) - len(missing_dates))

        # 2. Determine target dates for missing files
        updates_to_apply = {}
        for fpath in missing_dates:
            target_date = calculate_target_date(fpath)
            if target_date:
                updates_to_apply[fpath] = target_date
            else:
                folder_mismatch_skipped += 1

        # 3. Apply dates in batch
        if updates_to_apply:
            written = apply_dates_bulk(updates_to_apply)
            fixed_total += written

            sample_file = os.path.basename(list(updates_to_apply.keys())[0])
            sample_date = list(updates_to_apply.values())[0]
            extra = f" (+{written - 1} more)" if written > 1 else ""

            if written > 0:
                print(f"[{min(i + len(chunk), total)}/{total}] [FIXED {written} FILES] {sample_file} -> {sample_date}{extra}")
            else:
                print(f"[{min(i + len(chunk), total)}/{total}] [WRITE FAILED for {len(updates_to_apply)} FILES] {sample_file}")

        scanned_total += len(chunk)

        # Periodic progress summary
        if scanned_total % 1000 == 0 or scanned_total == total:
            elapsed = time.time() - t_start
            rate = scanned_total / elapsed if elapsed > 0 else 0
            print(f"--> Progress: {scanned_total}/{total} ({rate:.1f} files/s) | Dates fixed: {fixed_total}")

    print("\n" + "=" * 65)
    print(f"[COMPLETE] Date Repair Finished in {time.time() - t_start:.2f}s:")
    print(f"  Total Files Checked          : {scanned_total}")
    print(f"  Already Had Date Taken       : {has_date_skipped}")
    print(f"  Skipped (Folder Not YYYY/MM) : {folder_mismatch_skipped}")
    print(f"  Dates Successfully Written   : {fixed_total}")
    print("=" * 65)

if __name__ == "__main__":
    main()
