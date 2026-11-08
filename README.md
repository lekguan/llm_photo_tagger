# LLM Photo Tagger

This repo was created to use VLM to add tags to photos to help photo searching e.g. in Synology photos. This tool talks directly to the native **Synology Photos Web API** (`SYNO.Foto` / `SYNO.FotoTeam`). Tags appear immediately in the web interface and mobile app without requiring index rebuilds. The code/parameters are optimised for a web API which is the primary bottleneck for tagging speed, and it uses a prefix to ensure easy future removal/update without affecting other tags. It works with any vision-capable model with openAI compatible API (like `llama-server` or `vLLM`), but Qwen3 VL was chosen here for best tradeoff between speed and quality. 

---

## Key Features

- **Direct Synology Photos Web API Integration:** Interacts directly with `SYNO.Foto.Browse.Item`, `GeneralTag`, and `Thumbnail` endpoints over HTTP/HTTPS.
- **Pipelined Multi-Stage Concurrency:**
  - **1 Thumbnail Streamer:** Fetches lightweight, pre-generated `xl` thumbnails directly from DSM using authenticated sessions.
  - **Concurrent VLM Pool (`INFERENCE_WORKERS`):** Asynchronously dispatches thumbnail batches to your local vision model server.
  - **Concurrent NAS Writers (`NAS_WRITERS`):** Concurrent, verified tag application in `skip` mode with per-thread HTTP session isolation.
- **Thread-Safe Tag Catalog Management:** Centralized tag creation and ID resolution with re-entrant locking (`RLock`) to prevent duplicate tag creation on DSM.
- **Strict Read-Back Verification:** Polls the photo metadata after applying or clearing tags to confirm DSM committed the changes before marking an item as successful.
- **Three Operational Modes:**
  - `skip`: Scans library and tags only untagged items (skips photos already containing `TAG_PREFIX`).
  - `overwrite`: Re-analyzes photos, strips older prefix-generated tags, and applies fresh predictions while preserving custom manual tags.
  - `clear`: Fast maintenance mode that strips all prefix-generated tags across the library without running model inference.
- **Namespace Tag Isolation:** Uses a configurable prefix (default: `pt_`) to distinguish AI tags from manual or existing user tags.
- **Personal and Shared Space Support:** Easily toggle between personal libraries and Synology Shared Space (`USE_SHARED_SPACE`).

---

## Architecture Overview

```text
       ┌────────────────────────┐
       │ Synology Photos API    │ (Fetch candidate list)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ 1x Thumbnail Fetcher   │ (Downloads base64 'xl' thumbs)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  4x VLM Workers        │ (OpenAI-compatible /chat/completions)
       └───────────┬────────────┘
                   │ Bounded Queue (PREPARE_QUEUE_SIZE)
                   ▼
       ┌────────────────────────┐
       │  2x NAS Writers        │ (Resolve/Create Tag IDs -> Add Tag -> Verify)
       └────────────────────────┘
```

---

## Prerequisites

### 1. Environment Requirements
- Python 3.10+
- Network access to:
  - Your Synology NAS DSM web port (default: `5000` for HTTP or `5001` for HTTPS)
  - Your local VLM inference endpoint (default: `http://127.0.0.1:9931`)

### 2. Python Dependencies
Install the required standard dependencies:

```bash
pip install -r requirements.txt
```

---

## Local VLM Server Setup

The script expects an OpenAI-compatible vision completion endpoint (`/v1/chat/completions`).

### Example: Running with `llama-server`
### Example Launch Command for Vision Model
```bash
llama-server -m Qwen3-VL-8B-Instruct-Q5_K_M.gguf --mmproj /home/lek/llama_cpp/mmproj-Qwen3-VL-8B-F16.gguf \
  -ngl 99 --threads 4 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -np 4 -c 8192 -b 4096 -ub 2048 \
  --temp 0.2 --top-p 0.95 --min-p 0.05 --no-cache-prompt --cache-ram 0 --image-min-tokens 1024 --image-max-tokens 1024
```

---

## Configuration

Edit the settings block at the top of `tag_photos.py`:

```python
# ======================================================================
# NAS Credentials & Endpoint
# ======================================================================
NAS_HOST = "http://192.168.1.100:5000"  # Synology DSM URL
DSM_USERNAME = "your_dsm_username"
DSM_PASSWORD = "your_dsm_password"
USE_SHARED_SPACE = False  # Set to True for Shared Space (/photo)

# ======================================================================
# Model & Endpoint Configuration
# ======================================================================
SERVER_URL = "http://127.0.0.1:9931/v1/chat/completions"
VLM_MODEL = "qwen3-vl-8b"

# ======================================================================
# Execution Mode & Limits
# ======================================================================
# "skip"      -> Tag untagged photos only
# "overwrite" -> Re-analyze photos, replace old prefix tags
# "clear"     -> Strip all prefix tags without running VLM
MODE = "skip"

# Maximum photos to process (set to None for full library)
MAX_CANDIDATES = 100

# Prefix attached to all generated tags
TAG_PREFIX = "pt_"

# ======================================================================
# Concurrency & Tuning
# ======================================================================
INFERENCE_WORKERS = 4  # Concurrent requests to VLM server
NAS_WRITERS = 2  # Concurrent writers to Synology API (skip mode)
PREPARE_QUEUE_SIZE = 8  # Bounded queue buffer size
```

---

## Usage

Run the script directly from your terminal:

```bash
python tag_photos.py
```

### Execution Flow
1. **Login & Scan:** Authenticates against `SYNO.API.Auth` and pages through `Item.list` to collect eligible photos.
2. **Confirmation (for `overwrite` / `clear`):** If destructive actions are requested, the script requires typing `yes` before proceeding.
3. **Pipeline Processing:**
   - Fetches cached thumbnails (`Thumbnail.get`).
   - Sends the base64-encoded image to the local VLM with strict noun-focused prompts.
   - Normalizes tags and resolves catalog tag IDs (`GeneralTag.list` / `GeneralTag.create`).
   - Applies tags (`Item.add_tag`) and verifies via read-back (`Item.get`).
4. **Summary & Graceful Shutdown:** Outputs processed rates, skips, and verification status. Pressing `Ctrl+C` initiates safe backpressure shutdown, allowing in-flight writes to cleanly complete and verify.

---

## Tuning Recommendations

- **Writer Concurrency (`NAS_WRITERS`):** In testing against Synology DSM, `2` writers typically yields the best balance between speed and minimal DSM request contention. `overwrite` and `clear` modes automatically restrict themselves to `1` writer for tag preservation safety.
- **Inference Workers (`INFERENCE_WORKERS`):** Set this to match the parallel continuous batching slots configured on your local model runner (e.g. `-np 4` on `llama-server`).

