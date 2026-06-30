#!/usr/bin/env python3
"""
Code completion server with model management UI.

Manages llama.cpp server as a subprocess and provides:
- Web UI for downloading/managing GGUF models from HuggingFace
- Proxies /v1/* and native llama.cpp endpoints to the llama.cpp server
"""

import hmac
import html
import http.client
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

# Model weights default to temp/scratch storage (excluded from backups);
# small persistent state stays in app_data. start.sh sets these explicitly.
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/app_temp_data/code-completion/models"))
STATE_FILE = Path(os.environ.get("STATE_FILE", "/data/app_data/code-completion/state.json"))
# OpenHost injects the app's subdomain label and zone domain; together they
# form the public URL the app is served at, e.g.
# https://code-completion.user.host.imbue.com
OPENHOST_APP_NAME = os.environ.get("OPENHOST_APP_NAME", "")
OPENHOST_ZONE_DOMAIN = os.environ.get("OPENHOST_ZONE_DOMAIN", "")

N_THREADS = os.environ.get("N_THREADS", "4")
CTX_SIZE = os.environ.get("CTX_SIZE", "4096")
N_SLOTS = os.environ.get("N_SLOTS", "2")
GPU_LAYERS = os.environ.get("GPU_LAYERS", "0")

LLAMA_PORT = 8081
LLAMA_HOST = "127.0.0.1"

# Paths to proxy directly to llama-server
LLAMA_PROXY_PREFIXES = ("/v1/", "/infill", "/completions", "/tokenize", "/detokenize", "/embedding", "/slots")

# Global state
llama_process = None
llama_lock = threading.Lock()
downloads = {}  # model_file -> {progress, status, error}
download_lock = threading.Lock()

# Metrics: ring buffer of recent requests
MAX_METRICS = 1000
metrics_lock = threading.Lock()
request_metrics: list[dict] = []  # [{timestamp, latency, tokens, status, path}]
server_start_time = time.time()


# --- State management ---

state_lock = threading.RLock()


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(STATE_FILE)


def get_active_model():
    return load_state().get("active_model")


def set_active_model(model_file):
    with state_lock:
        state = load_state()
        state["active_model"] = model_file
        save_state(state)


# --- API token management ---
#
# The inference endpoints are exposed publicly (see public_paths in
# openhost.toml) so editors/tools can reach them without an OpenHost login.
# They are gated by a token generated and owned by this app (not the OpenHost
# API token). The owner, browsing through the router, bypasses the token check
# via the trusted X-OpenHost-Is-Owner header.

def get_api_token():
    """Return the app's API token, generating and persisting one on first use."""
    with state_lock:
        state = load_state()
        token = state.get("api_token")
        if not token:
            token = secrets.token_urlsafe(32)
            state["api_token"] = token
            save_state(state)
        return token


def regenerate_api_token():
    with state_lock:
        state = load_state()
        token = secrets.token_urlsafe(32)
        state["api_token"] = token
        save_state(state)
        return token


def extract_request_token(handler):
    """Pull a bearer/api-key token out of the request headers."""
    auth = handler.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    api_key = handler.headers.get("X-API-Key", "")
    if api_key:
        return api_key.strip()
    return None


def get_base_url(handler=None):
    """Best-effort public base URL the app is served at (no trailing slash).

    Prefers the router-supplied X-Forwarded-Host/Proto on the current request,
    then falls back to OPENHOST_APP_NAME + OPENHOST_ZONE_DOMAIN. Returns an
    empty string if the host can't be determined (local dev without env vars).
    """
    if handler is not None:
        host = handler.headers.get("X-Forwarded-Host", "")
        if host:
            proto = handler.headers.get("X-Forwarded-Proto", "https")
            return f"{proto}://{host}"
    if OPENHOST_APP_NAME and OPENHOST_ZONE_DOMAIN:
        return f"https://{OPENHOST_APP_NAME}.{OPENHOST_ZONE_DOMAIN}"
    return ""


def is_owner_request(handler):
    """The router sets X-OpenHost-Is-Owner: true for authenticated owner requests.

    Inbound X-OpenHost-* headers are stripped by the router, so this cannot be
    forged by external clients.
    """
    return handler.headers.get("X-OpenHost-Is-Owner", "").lower() == "true"


def is_authorized_inference(handler):
    """Authorize a request to a public inference endpoint."""
    if is_owner_request(handler):
        return True
    presented = extract_request_token(handler)
    if not presented:
        return False
    return hmac.compare_digest(presented, get_api_token())


# --- Model management ---

def list_models():
    if not MODELS_DIR.exists():
        return []
    models = []
    for f in sorted(MODELS_DIR.iterdir()):
        if f.suffix == ".gguf" and f.is_file():
            size_gb = f.stat().st_size / (1024 ** 3)
            models.append({
                "filename": f.name,
                "size_gb": round(size_gb, 2),
                "path": str(f),
            })
    return models


def start_llama(model_file):
    global llama_process
    with llama_lock:
        stop_llama_locked()
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            return False, f"Model file not found: {model_file}"

        cmd = [
            "/app/llama-server",
            "--model", str(model_path),
            "--host", LLAMA_HOST,
            "--port", str(LLAMA_PORT),
            "--threads", N_THREADS,
            "--ctx-size", CTX_SIZE,
            "--parallel", N_SLOTS,
            "-ngl", GPU_LAYERS,
        ]
        # Only enable flash attention when using GPU
        if int(GPU_LAYERS) > 0:
            cmd.extend(["--flash-attn", "on"])
        print(f"Starting llama-server: {' '.join(cmd)}", flush=True)
        llama_process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        set_active_model(model_file)

        for _ in range(120):
            time.sleep(1)
            if llama_process.poll() is not None:
                return False, "llama-server exited unexpectedly"
            try:
                conn = http.client.HTTPConnection(LLAMA_HOST, LLAMA_PORT, timeout=2)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status == 200:
                    print("llama-server is healthy", flush=True)
                    return True, None
            except Exception:
                pass

        return False, "llama-server failed to become healthy within 120s"


def stop_llama():
    with llama_lock:
        stop_llama_locked()
    set_active_model(None)


def stop_llama_locked():
    global llama_process
    if llama_process is not None:
        print("Stopping llama-server...", flush=True)
        llama_process.terminate()
        try:
            llama_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llama_process.kill()
            llama_process.wait(timeout=5)
        llama_process = None
        time.sleep(0.5)


def is_llama_running():
    with llama_lock:
        return llama_process is not None and llama_process.poll() is None


# --- Downloads ---

def record_metric(latency, tokens, status, path):
    with metrics_lock:
        request_metrics.append({
            "ts": time.time(),
            "latency": round(latency, 3),
            "tokens": tokens,
            "status": status,
            "path": path,
        })
        # Trim to ring buffer size
        if len(request_metrics) > MAX_METRICS:
            del request_metrics[:len(request_metrics) - MAX_METRICS]


def get_metrics_summary():
    now = time.time()
    with metrics_lock:
        all_reqs = list(request_metrics)

    if not all_reqs:
        return {"total_requests": 0, "uptime_s": int(now - server_start_time)}

    # Time windows
    last_1m = [r for r in all_reqs if now - r["ts"] < 60]
    last_5m = [r for r in all_reqs if now - r["ts"] < 300]
    last_1h = [r for r in all_reqs if now - r["ts"] < 3600]

    def stats(reqs):
        if not reqs:
            return {"count": 0, "rps": 0, "p50": 0, "p90": 0, "mean": 0, "avg_tokens": 0, "errors": 0}
        lats = sorted(r["latency"] for r in reqs)
        toks = [r["tokens"] for r in reqs]
        errs = sum(1 for r in reqs if r["status"] != 200)
        span = max(reqs[-1]["ts"] - reqs[0]["ts"], 1)
        return {
            "count": len(reqs),
            "rps": round(len(reqs) / span, 2),
            "p50": round(lats[len(lats) // 2], 3),
            "p90": round(lats[int(len(lats) * 0.9)], 3),
            "mean": round(sum(lats) / len(lats), 3),
            "avg_tokens": round(sum(toks) / len(toks), 1) if toks else 0,
            "errors": errs,
        }

    # Latency histogram for last 5 minutes (buckets)
    buckets = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
    hist = [0] * (len(buckets) + 1)
    for r in last_5m:
        placed = False
        for i, b in enumerate(buckets):
            if r["latency"] <= b:
                hist[i] += 1
                placed = True
                break
        if not placed:
            hist[-1] += 1

    # Requests per minute for the last hour (60 buckets)
    rpm_buckets = [0] * 60
    for r in last_1h:
        age_min = int((now - r["ts"]) / 60)
        if 0 <= age_min < 60:
            rpm_buckets[59 - age_min] += 1

    # Latency over time (last 5 min, 30 buckets of 10s each)
    lat_over_time = []
    for i in range(30):
        t_start = now - (30 - i) * 10
        t_end = t_start + 10
        bucket_reqs = [r for r in last_5m if t_start <= r["ts"] < t_end]
        if bucket_reqs:
            lats = [r["latency"] for r in bucket_reqs]
            lat_over_time.append({"p50": round(sorted(lats)[len(lats) // 2], 3), "count": len(bucket_reqs)})
        else:
            lat_over_time.append({"p50": 0, "count": 0})

    return {
        "total_requests": len(all_reqs),
        "uptime_s": int(now - server_start_time),
        "last_1m": stats(last_1m),
        "last_5m": stats(last_5m),
        "last_1h": stats(last_1h),
        "latency_hist": {"buckets": [f"<={b}s" for b in buckets] + [f">{buckets[-1]}s"], "counts": hist},
        "rpm": rpm_buckets,
        "lat_over_time": lat_over_time,
    }


def download_model(repo, filename):
    key = filename
    with download_lock:
        if key in downloads and downloads[key]["status"] == "downloading":
            return False, "Already downloading"
        downloads[key] = {"progress": "starting", "status": "downloading", "error": None}

    def _download():
        try:
            print(f"Downloading {repo}/{filename}...", flush=True)
            result = subprocess.run(
                ["hf", "download", repo, filename, "--local-dir", str(MODELS_DIR)],
                capture_output=True, text=True, timeout=7200,
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Download failed"
                print(f"Download failed: {error_msg}", flush=True)
                with download_lock:
                    downloads[key]["status"] = "failed"
                    downloads[key]["error"] = error_msg
                return

            if not (MODELS_DIR / filename).exists():
                with download_lock:
                    downloads[key]["status"] = "failed"
                    downloads[key]["error"] = "File not found after download"
                return

            print(f"Download complete: {filename}", flush=True)
            with download_lock:
                downloads[key]["status"] = "complete"
                downloads[key]["progress"] = "done"
        except subprocess.TimeoutExpired:
            with download_lock:
                downloads[key]["status"] = "failed"
                downloads[key]["error"] = "Download timed out (2h limit)"
        except Exception as e:
            with download_lock:
                downloads[key]["status"] = "failed"
                downloads[key]["error"] = str(e)

    threading.Thread(target=_download, daemon=True).start()
    return True, None


def clear_download(filename):
    with download_lock:
        downloads.pop(filename, None)


# --- HuggingFace ---

def search_huggingface(query):
    try:
        url = f"https://huggingface.co/api/models?search={quote(query)}&filter=gguf&sort=downloads&direction=-1&limit=20"
        req = urllib.request.Request(url, headers={"User-Agent": "openhost-code-completion/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return [{"id": m.get("id", ""), "downloads": m.get("downloads", 0), "likes": m.get("likes", 0)} for m in data]
    except Exception as e:
        print(f"HuggingFace search error: {e}", flush=True)
        return []


def list_repo_files(repo_id):
    try:
        url = f"https://huggingface.co/api/models/{repo_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "openhost-code-completion/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return sorted(s.get("rfilename", "") for s in data.get("siblings", []) if s.get("rfilename", "").endswith(".gguf"))
    except Exception as e:
        print(f"Error listing repo files: {e}", flush=True)
        return []


# --- Proxy ---

def proxy_to_llama(method, path, headers, body=None):
    try:
        conn = http.client.HTTPConnection(LLAMA_HOST, LLAMA_PORT, timeout=300)
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        resp_body = resp.read()
        resp_headers = dict(resp.getheaders())
        conn.close()
        return resp.status, resp_headers, resp_body
    except Exception as e:
        return 502, {}, json.dumps({"error": f"Backend unavailable: {e}"}).encode()


def should_proxy(path):
    return any(path.startswith(p) or path == p for p in LLAMA_PROXY_PREFIXES)


def get_proxy_headers(handler):
    headers = {}
    for key in ("content-type", "authorization", "accept"):
        val = handler.headers.get(key)
        if val:
            headers[key] = val
    return headers


# --- Filename validation ---

def validate_filename(filename):
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return False
    return filename.endswith(".gguf")


# --- HTML rendering ---

def render_page(models, active_model, download_status, api_token, base_url="",
                search_results=None,
                repo_files=None, selected_repo=None, message=None, error=None):
    running = is_llama_running()
    llama_status = "running" if running else "stopped"
    status_class = "active" if running else "stopped"

    # Models table
    model_rows = ""
    for m in models:
        is_active = m["filename"] == active_model
        active_badge = ' <span class="badge active">ACTIVE</span>' if is_active else ""
        buttons = ""
        if is_active:
            buttons = '''<form method="POST" action="/models/stop" style="display:inline">
                <button type="submit" class="btn btn-sm btn-warn">Stop</button></form>'''
        else:
            buttons = f'''<form method="POST" action="/models/activate" style="display:inline">
                <input type="hidden" name="filename" value="{html.escape(m['filename'])}">
                <button type="submit" class="btn btn-sm btn-primary">Load</button></form>'''
        buttons += f''' <form method="POST" action="/models/delete" style="display:inline"
              onsubmit="return confirm('Delete {html.escape(m["filename"])}?')">
            <input type="hidden" name="filename" value="{html.escape(m['filename'])}">
            <button type="submit" class="btn btn-sm btn-danger">Delete</button></form>'''
        model_rows += f'<tr><td class="model-name">{html.escape(m["filename"])}{active_badge}</td><td>{m["size_gb"]} GB</td><td class="actions">{buttons}</td></tr>'

    # Downloads
    download_rows = ""
    for fname, info in download_status.items():
        sc = {"downloading": "downloading", "complete": "complete", "failed": "failed"}.get(info["status"], "")
        err = f' <span class="dl-error">{html.escape(info["error"])}</span>' if info.get("error") else ""
        dismiss = ""
        if info["status"] in ("complete", "failed"):
            dismiss = f''' <form method="POST" action="/downloads/clear" style="display:inline">
                <input type="hidden" name="filename" value="{html.escape(fname)}">
                <button type="submit" class="btn btn-sm btn-muted">Dismiss</button></form>'''
        download_rows += f'<tr><td>{html.escape(fname)}</td><td><span class="badge {sc}">{html.escape(info["status"])}</span>{err}</td><td>{dismiss}</td></tr>'

    # Search results
    search_html = ""
    if search_results is not None:
        if search_results:
            rows = "".join(f'<tr><td><a href="/?browse={html.escape(r["id"])}">{html.escape(r["id"])}</a></td><td>{r["downloads"]:,}</td><td>{r["likes"]:,}</td></tr>' for r in search_results)
            search_html = f'<table><tr><th>Repository</th><th>Downloads</th><th>Likes</th></tr>{rows}</table>'
        else:
            search_html = '<p class="empty">No results found.</p>'

    # Browse repo files
    browse_html = ""
    if repo_files is not None and selected_repo:
        if repo_files:
            rows = "".join(f'''<tr><td>{html.escape(f)}</td><td>
                <form method="POST" action="/models/download" style="display:inline">
                    <input type="hidden" name="repo" value="{html.escape(selected_repo)}">
                    <input type="hidden" name="filename" value="{html.escape(f)}">
                    <button type="submit" class="btn btn-sm btn-primary">Download</button>
                </form></td></tr>''' for f in repo_files)
            browse_html = f'<h3>Files in {html.escape(selected_repo)}</h3><table><tr><th>Filename</th><th></th></tr>{rows}</table>'
        else:
            browse_html = f'<p class="empty">No GGUF files found in {html.escape(selected_repo)}.</p>'

    # Messages
    msg_html = ""
    if message:
        msg_html = f'<div class="alert success">{html.escape(message)}</div>'
    if error:
        msg_html = f'<div class="alert error">{html.escape(error)}</div>'

    # Public base URL for setup snippets; fall back to a placeholder if unknown.
    display_url = base_url or "https://code-completion.<zone>"
    esc_url = html.escape(display_url)

    # API token section
    token_html = f"""
    <h2>API Token</h2>
    <div class="section">
        <p class="hint" style="margin-bottom:8px">Send this token as
        <code>Authorization: Bearer &lt;token&gt;</code> when calling the inference
        endpoints. Keep it secret &mdash; anyone with it can use this server.</p>
        <div class="form-row">
            <input type="text" id="api-token" value="{html.escape(api_token)}" readonly
                   onclick="this.select()">
            <button type="button" class="btn" onclick="copyToken()">Copy</button>
            <form method="POST" action="/token/regenerate" style="display:inline"
                  onsubmit="return confirm('Regenerate the token? Existing clients will stop working until updated.')">
                <button type="submit" class="btn btn-warn">Regenerate</button>
            </form>
        </div>
    </div>"""

    # Auto-refresh if downloads are active
    has_active_downloads = any(d["status"] == "downloading" for d in download_status.values())
    meta_refresh = '<meta http-equiv="refresh" content="3">' if has_active_downloads else ""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Code Completion</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {meta_refresh}
    <style>
        :root {{ --bg: #0d1117; --surface: #161b22; --border: #21262d; --text: #c9d1d9;
                 --text-dim: #8b949e; --blue: #58a6ff; --green: #238636; --green-h: #2ea043;
                 --red: #da3633; --red-h: #f85149; --yellow: #d29922; --yellow-bg: #2d2000; }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               background: var(--bg); color: var(--text); line-height: 1.6; }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 24px 20px; }}
        h1 {{ color: var(--blue); font-size: 1.4em; margin-bottom: 16px; }}
        h2 {{ color: var(--text); font-size: 1.1em; margin: 28px 0 12px;
              border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
        h3 {{ color: var(--text); font-size: 0.95em; margin: 16px 0 8px; }}
        a {{ color: var(--blue); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        table {{ width: 100%; border-collapse: collapse; margin: 8px 0 12px; }}
        th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
        th {{ color: var(--text-dim); font-weight: 600; font-size: 0.85em; text-transform: uppercase;
              letter-spacing: 0.05em; }}
        td.model-name {{ font-family: monospace; font-size: 0.9em; }}
        td.actions {{ white-space: nowrap; }}
        .btn {{ display: inline-block; border: none; padding: 6px 14px; border-radius: 6px;
                cursor: pointer; font-size: 13px; font-weight: 500; color: #fff;
                background: var(--green); transition: background 0.15s; }}
        .btn:hover {{ background: var(--green-h); }}
        .btn-sm {{ padding: 4px 10px; font-size: 12px; }}
        .btn-primary {{ background: var(--green); }}
        .btn-primary:hover {{ background: var(--green-h); }}
        .btn-danger {{ background: var(--red); }}
        .btn-danger:hover {{ background: var(--red-h); }}
        .btn-warn {{ background: var(--yellow); color: #000; }}
        .btn-warn:hover {{ background: #e5a523; }}
        .btn-muted {{ background: var(--border); color: var(--text-dim); }}
        .btn-muted:hover {{ background: #30363d; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
                  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }}
        .badge.active {{ background: var(--green); color: #fff; }}
        .badge.stopped {{ background: var(--border); color: var(--text-dim); }}
        .badge.downloading {{ background: #1f6feb; color: #fff; }}
        .badge.complete {{ background: var(--green); color: #fff; }}
        .badge.failed {{ background: var(--red); color: #fff; }}
        input[type="text"] {{ background: var(--bg); border: 1px solid var(--border); color: var(--text);
                              padding: 7px 12px; border-radius: 6px; font-size: 14px; width: 100%; }}
        input[type="text"]:focus {{ outline: none; border-color: var(--blue); }}
        .form-row {{ display: flex; gap: 8px; align-items: center; margin: 8px 0; }}
        .form-row input {{ flex: 1; }}
        .alert {{ padding: 10px 14px; border-radius: 6px; margin: 12px 0; font-size: 0.9em; }}
        .alert.success {{ background: #0d2818; border: 1px solid var(--green); }}
        .alert.error {{ background: #2d1117; border: 1px solid var(--red); color: var(--red-h); }}
        .status-bar {{ background: var(--surface); padding: 12px 16px; border-radius: 8px;
                       margin-bottom: 20px; display: flex; gap: 20px; align-items: center;
                       flex-wrap: wrap; border: 1px solid var(--border); }}
        .status-bar .label {{ color: var(--text-dim); font-size: 0.85em; }}
        .status-bar .value {{ font-weight: 600; }}
        .section {{ background: var(--surface); padding: 16px; border-radius: 8px;
                    margin: 8px 0 16px; border: 1px solid var(--border); }}
        .hint {{ color: var(--text-dim); font-size: 0.8em; margin-top: 4px; }}
        .empty {{ color: var(--text-dim); font-style: italic; padding: 8px 0; }}
        .dl-error {{ color: var(--red-h); font-size: 0.85em; }}
        code {{ background: var(--bg); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
        pre {{ background: var(--bg); padding: 12px; border-radius: 6px; margin: 8px 0;
               overflow-x: auto; font-size: 0.85em; line-height: 1.5; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
        @media (max-width: 600px) {{ .two-col {{ grid-template-columns: 1fr; }}
            .form-row {{ flex-direction: column; }} .form-row input {{ width: 100%; }} }}
    </style>
</head>
<body>
<div class="container">
    <h1>Code Completion</h1>
    <div style="margin-bottom:12px;font-size:0.9em"><a href="/">Models</a> &nbsp; <a href="/dashboard">Dashboard</a></div>

    <div class="status-bar">
        <div><span class="label">Server</span><br><span class="badge {status_class}">{llama_status}</span></div>
        <div><span class="label">Model</span><br><span class="value">{html.escape(active_model or 'none')}</span></div>
        <div><span class="label">Endpoint</span><br><code>/infill</code></div>
        {"" if not running else '<div style="margin-left:auto"><form method="POST" action="/models/stop"><button type="submit" class="btn btn-sm btn-warn">Stop Server</button></form></div>'}
    </div>

    {msg_html}

    {token_html}

    <h2>Models</h2>
    <div class="section">
    {"<table><tr><th>Model</th><th>Size</th><th></th></tr>" + model_rows + "</table>" if models else '<p class="empty">No models downloaded yet. Search HuggingFace below to get started.</p>'}
    </div>

    {"<h2>Downloads</h2><div class='section'><table><tr><th>File</th><th>Status</th><th></th></tr>" + download_rows + "</table></div>" if download_rows else ""}

    <h2>Get Models</h2>
    <div class="section">
        <form method="GET" action="/">
            <div class="form-row">
                <input type="text" name="search" placeholder="Search HuggingFace (e.g. qwen2.5-coder gguf)" value="">
                <button type="submit" class="btn">Search</button>
            </div>
        </form>

        {search_html}
        {browse_html}

        <details style="margin-top:16px">
            <summary style="cursor:pointer;color:var(--text-dim);font-size:0.9em">Direct download (paste repo + filename)</summary>
            <form method="POST" action="/models/download" style="margin-top:8px">
                <div class="form-row">
                    <input type="text" name="repo" placeholder="owner/repo">
                    <input type="text" name="filename" placeholder="model.gguf">
                    <button type="submit" class="btn">Download</button>
                </div>
            </form>
        </details>
    </div>

    <details style="margin-top:24px">
        <summary style="cursor:pointer;color:var(--text-dim)">Setup instructions</summary>
        <div class="section" style="margin-top:8px">
            <h3>Neovim (llama.vim)</h3>
            <pre>vim.g.llama_config = {{
  endpoint_fim = "{esc_url}/infill",
  api_key = "{html.escape(api_token)}",
  n_predict = 128,
  t_max_predict_ms = 5000,
}}</pre>
            <h3>VSCodium (llama-vscode)</h3>
            <pre>{{
  "llama.endpoint": "{esc_url}",
  "llama.api_key": "{html.escape(api_token)}"
}}</pre>
            <h3>OpenAI-compatible API</h3>
            <pre>curl {esc_url}/v1/completions \\
  -H "Authorization: Bearer {html.escape(api_token)}" \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt": "def fib(n):", "n_predict": 64}}'

POST {esc_url}/v1/completions
POST {esc_url}/v1/chat/completions</pre>
        </div>
    </details>
</div>
<script>
function copyToken() {{
    var el = document.getElementById('api-token');
    el.select();
    navigator.clipboard.writeText(el.value);
}}
</script>
</body>
</html>"""


def render_dashboard():
    m = get_metrics_summary()
    uptime_h = m["uptime_s"] // 3600
    uptime_m = (m["uptime_s"] % 3600) // 60

    active = get_active_model() or "none"
    running = is_llama_running()

    # SVG bar chart helper
    def bar_chart(values, labels, width=500, height=120, color="#58a6ff"):
        if not values or max(values) == 0:
            return f'<svg width="{width}" height="{height}"><text x="10" y="60" fill="#8b949e">No data</text></svg>'
        max_val = max(values)
        n = len(values)
        bar_w = max((width - 40) // n - 1, 2)
        bars = ""
        for i, v in enumerate(values):
            h = int((v / max_val) * (height - 30)) if max_val > 0 else 0
            x = 30 + i * (bar_w + 1)
            y = height - 20 - h
            bars += f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" rx="1"/>'
        # Y axis labels
        bars += f'<text x="0" y="15" fill="#8b949e" font-size="10">{max_val}</text>'
        bars += f'<text x="0" y="{height - 20}" fill="#8b949e" font-size="10">0</text>'
        # X axis labels (first, middle, last)
        if labels:
            bars += f'<text x="30" y="{height - 4}" fill="#8b949e" font-size="9">{labels[0]}</text>'
            if len(labels) > 1:
                bars += f'<text x="{width - 40}" y="{height - 4}" fill="#8b949e" font-size="9" text-anchor="end">{labels[-1]}</text>'
        return f'<svg width="{width}" height="{height}" style="display:block">{bars}</svg>'

    # RPM chart (last hour, per minute)
    rpm = m.get("rpm", [])
    rpm_labels = ["-60m", "now"]
    rpm_chart = bar_chart(rpm, rpm_labels, width=600, height=100, color="#238636")

    # Latency over time (last 5 min, 10s buckets)
    lot = m.get("lat_over_time", [])
    lot_vals = [x["p50"] * 1000 for x in lot]  # ms
    lot_labels = ["-5m", "now"]
    lot_chart = bar_chart(lot_vals, lot_labels, width=600, height=100, color="#d29922")

    # Latency histogram
    hist = m.get("latency_hist", {})
    hist_vals = hist.get("counts", [])
    hist_labels = hist.get("buckets", [])
    hist_chart = bar_chart(hist_vals, [hist_labels[0] if hist_labels else "", hist_labels[-1] if hist_labels else ""], width=600, height=100, color="#58a6ff")

    def stat_card(label, value, sub=""):
        sub_html = f'<div style="color:var(--text-dim);font-size:0.8em">{sub}</div>' if sub else ""
        return f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">{value}</div>{sub_html}</div>'

    s1m = m.get("last_1m", {})
    s5m = m.get("last_5m", {})
    s1h = m.get("last_1h", {})

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dashboard - Code Completion</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="5">
    <style>
        :root {{ --bg: #0d1117; --surface: #161b22; --border: #21262d; --text: #c9d1d9;
                 --text-dim: #8b949e; --blue: #58a6ff; --green: #238636; }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               background: var(--bg); color: var(--text); line-height: 1.6; }}
        .container {{ max-width: 960px; margin: 0 auto; padding: 24px 20px; }}
        h1 {{ color: var(--blue); font-size: 1.4em; margin-bottom: 4px; }}
        h2 {{ color: var(--text); font-size: 1.05em; margin: 20px 0 8px;
              border-bottom: 1px solid var(--border); padding-bottom: 4px; }}
        a {{ color: var(--blue); text-decoration: none; }}
        .nav {{ margin-bottom: 16px; font-size: 0.9em; }}
        .nav a {{ margin-right: 16px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 10px 0; }}
        .stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
        .stat-label {{ color: var(--text-dim); font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.05em; }}
        .stat-value {{ font-size: 1.4em; font-weight: 700; margin-top: 2px; }}
        .chart-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin: 10px 0; overflow-x: auto; }}
        .chart-title {{ color: var(--text-dim); font-size: 0.8em; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
        .badge.active {{ background: var(--green); color: #fff; }}
        .badge.stopped {{ background: var(--border); color: var(--text-dim); }}
        code {{ font-size: 0.85em; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Dashboard</h1>
    <div class="nav"><a href="/">Models</a> <a href="/dashboard">Dashboard</a></div>

    <div class="stat-grid">
        {stat_card("Status", f'<span class="badge {"active" if running else "stopped"}">{"running" if running else "stopped"}</span>')}
        {stat_card("Model", f'<code>{html.escape(active)}</code>')}
        {stat_card("Uptime", f'{uptime_h}h {uptime_m}m')}
        {stat_card("Total Requests", str(m["total_requests"]))}
    </div>

    <h2>Last 1 Minute</h2>
    <div class="stat-grid">
        {stat_card("Requests", str(s1m.get("count", 0)))}
        {stat_card("RPS", str(s1m.get("rps", 0)))}
        {stat_card("p50 Latency", f'{s1m.get("p50", 0):.3f}s')}
        {stat_card("p90 Latency", f'{s1m.get("p90", 0):.3f}s')}
        {stat_card("Mean Latency", f'{s1m.get("mean", 0):.3f}s')}
        {stat_card("Avg Tokens", str(s1m.get("avg_tokens", 0)))}
        {stat_card("Errors", str(s1m.get("errors", 0)))}
    </div>

    <h2>Last 5 Minutes</h2>
    <div class="stat-grid">
        {stat_card("Requests", str(s5m.get("count", 0)))}
        {stat_card("RPS", str(s5m.get("rps", 0)))}
        {stat_card("p50 Latency", f'{s5m.get("p50", 0):.3f}s')}
        {stat_card("p90 Latency", f'{s5m.get("p90", 0):.3f}s')}
        {stat_card("Errors", str(s5m.get("errors", 0)))}
    </div>

    <h2>Last Hour</h2>
    <div class="stat-grid">
        {stat_card("Requests", str(s1h.get("count", 0)))}
        {stat_card("p50 Latency", f'{s1h.get("p50", 0):.3f}s')}
        {stat_card("p90 Latency", f'{s1h.get("p90", 0):.3f}s')}
        {stat_card("Errors", str(s1h.get("errors", 0)))}
    </div>

    <div class="chart-box">
        <div class="chart-title">Requests per Minute (last hour)</div>
        {rpm_chart}
    </div>

    <div class="chart-box">
        <div class="chart-title">p50 Latency (ms) over last 5 minutes (10s buckets)</div>
        {lot_chart}
    </div>

    <div class="chart-box">
        <div class="chart-title">Latency Distribution (last 5 minutes)</div>
        {hist_chart}
    </div>
</div>
</body>
</html>"""


# --- HTTP Handler ---

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/health":
            self._json(200, {"status": "ok" if is_llama_running() else "no_model_loaded"})
            return

        if should_proxy(parsed.path):
            self._proxy_or_503("GET", parsed.path)
            return

        if parsed.path == "/api/models":
            self._json(200, {"models": list_models(), "active_model": get_active_model(), "downloads": dict(downloads)})
            return

        if parsed.path == "/api/metrics":
            self._json(200, get_metrics_summary())
            return

        if parsed.path == "/dashboard":
            self._html(200, render_dashboard())
            return

        if parsed.path == "/":
            search_results = search_huggingface(params["search"][0]) if "search" in params else None
            repo_files = list_repo_files(params["browse"][0]) if "browse" in params else None
            selected_repo = params.get("browse", [None])[0]
            body = render_page(
                models=list_models(), active_model=get_active_model() or "",
                download_status=dict(downloads), api_token=get_api_token(),
                base_url=get_base_url(self),
                search_results=search_results,
                repo_files=repo_files, selected_repo=selected_repo,
                message=params.get("msg", [None])[0], error=params.get("err", [None])[0],
            )
            self._html(200, body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        if should_proxy(parsed.path):
            self._proxy_or_503("POST", parsed.path, body)
            return

        form_data = self._parse_form(body)
        get_field = lambda name: (form_data.get(name, [""])[0] if isinstance(form_data.get(name), list) else form_data.get(name, "")).strip()

        if parsed.path == "/models/download":
            repo, filename = get_field("repo"), get_field("filename")
            if not repo or not filename:
                return self._redirect_err("Both repo and filename are required.")
            if not validate_filename(filename):
                return self._redirect_err("Invalid filename. Must be a .gguf file.")
            ok, err = download_model(repo, filename)
            return self._redirect_msg(f"Downloading {filename}...") if ok else self._redirect_err(err or "Download failed")

        if parsed.path == "/models/activate":
            filename = get_field("filename")
            if not filename or not validate_filename(filename):
                return self._redirect_err("Invalid model.")
            if not (MODELS_DIR / filename).exists():
                return self._redirect_err(f"Model not found: {filename}")
            ok, err = start_llama(filename)
            return self._redirect_msg(f"Loaded {filename}") if ok else self._redirect_err(err or "Failed to load model")

        if parsed.path == "/models/stop":
            stop_llama()
            return self._redirect_msg("Server stopped.")

        if parsed.path == "/models/delete":
            filename = get_field("filename")
            if not filename or not validate_filename(filename):
                return self._redirect_err("Invalid filename.")
            if filename == get_active_model():
                stop_llama()
            model_path = MODELS_DIR / filename
            if model_path.exists():
                model_path.unlink()
            clear_download(filename)
            return self._redirect_msg(f"Deleted {filename}")

        if parsed.path == "/downloads/clear":
            filename = get_field("filename")
            if filename:
                clear_download(filename)
            return self._redirect_msg("Cleared.")

        if parsed.path == "/token/regenerate":
            regenerate_api_token()
            return self._redirect_msg("API token regenerated.")

        self.send_response(404)
        self.end_headers()

    # --- Helpers ---

    def _proxy_or_503(self, method, path, body=None):
        if not is_authorized_inference(self):
            self._json(401, {"error": {
                "message": "Missing or invalid API token. Provide it as "
                           "'Authorization: Bearer <token>'.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }})
            return
        if not is_llama_running():
            self._json(503, {"error": {"message": "No model loaded.", "type": "server_error"}})
            return
        headers = get_proxy_headers(self)
        start = time.time()
        status, resp_headers, resp_body = proxy_to_llama(method, path, headers, body)
        latency = time.time() - start

        # Record metrics for inference endpoints
        tokens = 0
        if status == 200 and path in ("/infill", "/completions") or path.startswith("/v1/"):
            try:
                data = json.loads(resp_body)
                tokens = data.get("tokens_predicted", 0) or data.get("usage", {}).get("completion_tokens", 0)
            except Exception:
                pass
            record_metric(latency, tokens, status, path)

        self.send_response(status)
        for k, v in resp_headers.items():
            if k.lower() not in ("transfer-encoding", "connection"):
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(resp_body)

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _html(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def _redirect_msg(self, msg):
        self.send_response(303)
        self.send_header("Location", f"/?msg={quote(msg)}")
        self.end_headers()

    def _redirect_err(self, err):
        self.send_response(303)
        self.send_header("Location", f"/?err={quote(err)}")
        self.end_headers()

    def _parse_form(self, body):
        ct = self.headers.get("Content-Type", "")
        if "application/json" in ct:
            return json.loads(body) if body else {}
        return parse_qs(body.decode())

    def log_message(self, format, *args):
        print(f"[HTTP] {args[0]}", flush=True)


def main():
    active = get_active_model()
    if active and (MODELS_DIR / active).exists():
        print(f"Auto-starting previously active model: {active}", flush=True)
        ok, err = start_llama(active)
        if not ok:
            print(f"Failed to auto-start model: {err}", flush=True)

    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("Web UI listening on :8080", flush=True)

    def handle_signal(signum, frame):
        print(f"Received signal {signum}, shutting down...", flush=True)
        with llama_lock:
            stop_llama_locked()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    server.serve_forever()


if __name__ == "__main__":
    main()
