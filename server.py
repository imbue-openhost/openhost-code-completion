#!/usr/bin/env python3
"""
Code completion server with model management UI.

Manages llama.cpp server as a subprocess and provides:
- Web UI for downloading/managing GGUF models from HuggingFace
- Proxies /v1/* and native llama.cpp endpoints to the llama.cpp server
"""

import html
import http.client
import json
import os
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

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/app_data/code-completion/models"))
STATE_FILE = Path(os.environ.get("STATE_FILE", "/data/app_data/code-completion/state.json"))
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


# --- State management ---

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
    state = load_state()
    state["active_model"] = model_file
    save_state(state)


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
            "--flash-attn",
        ]
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

def render_page(models, active_model, download_status, search_results=None,
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

    <div class="status-bar">
        <div><span class="label">Server</span><br><span class="badge {status_class}">{llama_status}</span></div>
        <div><span class="label">Model</span><br><span class="value">{html.escape(active_model or 'none')}</span></div>
        <div><span class="label">Endpoint</span><br><code>/infill</code></div>
        {"" if not running else '<div style="margin-left:auto"><form method="POST" action="/models/stop"><button type="submit" class="btn btn-sm btn-warn">Stop Server</button></form></div>'}
    </div>

    {msg_html}

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
  endpoint_fim = "https://code-completion.&lt;zone&gt;/infill",
  api_key = "&lt;token&gt;",
  n_predict = 128,
  t_max_predict_ms = 5000,
}}</pre>
            <h3>VSCodium (llama-vscode)</h3>
            <pre>{{
  "llama.endpoint": "https://code-completion.&lt;zone&gt;",
  "llama.api_key": "&lt;token&gt;"
}}</pre>
            <h3>OpenAI-compatible API</h3>
            <pre>POST /v1/completions
POST /v1/chat/completions</pre>
        </div>
    </details>
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

        if parsed.path == "/":
            search_results = search_huggingface(params["search"][0]) if "search" in params else None
            repo_files = list_repo_files(params["browse"][0]) if "browse" in params else None
            selected_repo = params.get("browse", [None])[0]
            body = render_page(
                models=list_models(), active_model=get_active_model() or "",
                download_status=dict(downloads), search_results=search_results,
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

        self.send_response(404)
        self.end_headers()

    # --- Helpers ---

    def _proxy_or_503(self, method, path, body=None):
        if not is_llama_running():
            self._json(503, {"error": {"message": "No model loaded.", "type": "server_error"}})
            return
        headers = get_proxy_headers(self)
        status, resp_headers, resp_body = proxy_to_llama(method, path, headers, body)
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
