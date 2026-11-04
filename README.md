# LLM Photo Tagger

This repo was created to use VLM to add tags to photos to help photo searching e.g. in Synology photos. The main feature in this repo is that the code/parameters are optimised for a mounted network drive which is the primary bottleneck for tagging speed, and it uses a prefix to ensure easy future removal/update without affecting other tags. It works with any vision-capable model with openAI compatible API, but Qwen3 VL was chosen here for best tradeoff between speed and quality.


## Architecture & Scripts

1. **`tag_photos.py` (Fast Photo Tagger)**
   - **Streaming Chunk Inspection:** Evaluates files in 250-item batches using ExifTool over stdin, eliminating per-file process startup overhead.
   - **In-Memory RAM Prefetcher:** Downloads and Lanczos-downsamples images into Base64 payloads ahead of time, ensuring 0 ms disk read latency when a GPU worker is ready.
   - **Decoupled Asynchronous Writer:** Offloads XMP/IPTC/EXIF file writes to a dedicated background worker so GPU workers never block on remote SMB/NFS disk write locks.
   - **Zero Sleep Dispatches:** Fully relies on `llama-server` continuous batching without artificial client-side delays.

2. **`strip_tags.py` (Fast Tag Stripper)**
   - **Batch ExifTool Cleaning:** Removes `st_` tags in batches of 100 using temporary argfiles.
   - **Stale Temp Cleanup:** Detects and removes leftover `*_exiftool_tmp` files left behind by aborted processes before writing.
   - **Accurate Auditing:** Verifies execution return codes to provide accurate counts of clean, skipped, and stripped files.

---

## Prerequisites

### 1. System Dependencies (ExifTool)
ExifTool is required on the host system to read and write metadata.

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y exiftool libimage-exiftool-perl

# Arch Linux
sudo pacman -S perl-image-exiftool

# RHEL / Fedora / Rocky Linux
sudo dnf install -y perl-Image-ExifTool
```

Verify installation:
```bash
exiftool -ver
```

### 2. Python Environment Setup
```bash
pip install -r requirements.txt
```

---

## Server Deployment

Run `llama-server`.

### Example Launch Command for Vision Model (Target Nvidia V100)
```bash
llama-server -m Qwen3-VL-8B-Instruct-Q5_K_M.gguf --mmproj mmproj-F16.gguf \
  -ngl 99 --threads 4 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -np 4 -c 8192 -b 4096 -ub 2048 \
  --temp 0.2 --top-p 0.95 --min-p 0.05 --no-cache-prompt --cache-ram 0 --image-min-tokens 1024 --image-max-tokens 1024
```

### Server Configuration Details
- `-np 4 -c 8192`: Allocates 4 parallel continuous batching slots with 2,048 tokens each (1,024 vision tokens + prompt + 128 response tokens).
- `-ub 2048`: Ensures the entire 1,024-token visual patch sequence evaluates in a single forward pass without chunking stalls.
- `--no-cache-prompt --cache-ram 0`: Disables prompt prefix caching to prevent PCIe/RAM thrashing from unique image payloads.

---

## Usage

### 1. Tagging Untagged Photos
Runs the streaming prefetch pipeline, sends requests to `llama-server`, and writes tags under the `st_` prefix:

```bash
python tag_photos.py
```

Configuration variables inside `tag_photos.py`:
- `PHOTO_DIR`: Path to your local or mounted photo library.
- `SERVER_URL`: `llama-server` endpoint (`http://127.0.0.1:9931/v1/chat/completions`).
- `TAG_PREFIX`: Custom tag prefix (`st_`) so it can be stripped easily in future if required while leaving other tags untouched.
- `MAX_WORKERS`: Set to `4` to match your `-np 4` server slots.
- `CHUNK_SIZE`: Number of files inspected per ExifTool batch check (`250`).

### 2. Stripping Generated Tags
Quickly removes all metadata tags starting with `st_` across your library:

```bash
python strip_tags.py
```

---

## Troubleshooting & Maintenance

- **Orphaned Temp Files (`*_exiftool_tmp`):**
  If a previous run was killed abruptly and left stale lock files on your network share, clear them manually:
  ```bash
  find "/path/to/photos" -type f -name "*_exiftool_tmp" -delete
  ```
- **Network Share Latency:**
  To optimise network drive speed, make sure asynchronous caching is enabled (e.g. `actimeo=30` on NFS or `cache=loose` on SMB). Example:
  ```bash
  sudo mount -t cifs //nas.local/homes /mnt/synology_photos -o username=username,uid=$(id -u),gid=$(id -g),iocharset=utf8,vers=3.1.1,cache=loose,actimeo=60
  ```
