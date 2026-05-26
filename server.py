#!/usr/bin/env python3
"""
Code completion server with model management UI.

Manages llama.cpp server as a subprocess and provides:
- Web UI for downloading/managing GGUF models from HuggingFace
- Proxies /v1/* requests to the llama.cpp server for OpenAI-compatible API
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

LLAMA_PORT = 8081
LLAMA_HOST = "127.0.0.1"

# Global state
llama_process = None
llama_lock = threading.Lock()
downloads = {}  # model_file -> {progress, status, error}
download_lock = threading.Lock()


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
    state = load_state()
    return state.get("active_model")


def set_active_model(model_file):
    state = load_state()
    state["active_model"] = model_file
    save_state(state)


def list_models():
    """Return list of downloaded model files."""
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
    """Start llama-server with the given model."""
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
        ]
        print(f"Starting llama-server: {' '.join(cmd)}", flush=True)
        llama_process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        set_active_model(model_file)

        # Wait for it to become healthy
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


def stop_llama_locked():
    """Stop llama-server. Must hold llama_lock."""
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


def download_model(repo, filename):
    """Download a model from HuggingFace in a background thread."""
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
                capture_output=True,
                text=True,
                timeout=7200,
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Download failed"
                print(f"Download failed: {error_msg}", flush=True)
                with download_lock:
                    downloads[key]["status"] = "failed"
                    downloads[key]["error"] = error_msg
                return

            model_path = MODELS_DIR / filename
            if not model_path.exists():
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

    t = threading.Thread(target=_download, daemon=True)
    t.start()
    return True, None


def search_huggingface(query):
    """Search HuggingFace for GGUF models."""
    try:
        url = f"https://huggingface.co/api/models?search={query}&filter=gguf&sort=downloads&direction=-1&limit=20"
        req = urllib.request.Request(url, headers={"User-Agent": "openhost-code-completion/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = []
        for model in data:
            results.append({
                "id": model.get("id", ""),
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
            })
        return results
    except Exception as e:
        print(f"HuggingFace search error: {e}", flush=True)
        return []


def list_repo_files(repo_id):
    """List GGUF files in a HuggingFace repo."""
    try:
        url = f"https://huggingface.co/api/models/{repo_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "openhost-code-completion/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        files = []
        for sib in data.get("siblings", []):
            fname = sib.get("rfilename", "")
            if fname.endswith(".gguf"):
                files.append(fname)
        return sorted(files)
    except Exception as e:
        print(f"Error listing repo files: {e}", flush=True)
        return []


def proxy_to_llama(method, path, headers, body=None):
    """Proxy a request to the llama-server."""
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


# HTML template
def render_page(models, active_model, download_status, search_results=None,
                repo_files=None, selected_repo=None, message=None, error=None):
    model_rows = ""
    for m in models:
        is_active = m["filename"] == active_model
        active_badge = '<span class="badge active">ACTIVE</span>' if is_active else ""
        activate_btn = "" if is_active else f'''
            <form method="POST" action="/models/activate" style="display:inline">
                <input type="hidden" name="filename" value="{html.escape(m['filename'])}">
                <button type="submit" class="btn btn-sm">Activate</button>
            </form>'''
        model_rows += f"""
        <tr>
            <td>{html.escape(m['filename'])} {active_badge}</td>
            <td>{m['size_gb']} GB</td>
            <td>
                {activate_btn}
                <form method="POST" action="/models/delete" style="display:inline"
                      onsubmit="return confirm('Delete {html.escape(m['filename'])}?')">
                    <input type="hidden" name="filename" value="{html.escape(m['filename'])}">
                    <button type="submit" class="btn btn-sm btn-danger">Delete</button>
                </form>
            </td>
        </tr>"""

    download_rows = ""
    for fname, info in download_status.items():
        status_class = "downloading" if info["status"] == "downloading" else (
            "complete" if info["status"] == "complete" else "failed")
        error_text = f' - {html.escape(info["error"])}' if info.get("error") else ""
        download_rows += f"""
        <tr>
            <td>{html.escape(fname)}</td>
            <td><span class="badge {status_class}">{html.escape(info['status'])}</span>{error_text}</td>
        </tr>"""

    search_html = ""
    if search_results is not None:
        if search_results:
            rows = ""
            for r in search_results:
                rows += f"""
                <tr>
                    <td>
                        <a href="/?browse={html.escape(r['id'])}">{html.escape(r['id'])}</a>
                    </td>
                    <td>{r['downloads']:,}</td>
                    <td>{r['likes']:,}</td>
                </tr>"""
            search_html = f"""
            <h3>Search Results</h3>
            <table>
                <tr><th>Repository</th><th>Downloads</th><th>Likes</th></tr>
                {rows}
            </table>"""
        else:
            search_html = "<p>No results found.</p>"

    browse_html = ""
    if repo_files is not None and selected_repo:
        if repo_files:
            rows = ""
            for fname in repo_files:
                rows += f"""
                <tr>
                    <td>{html.escape(fname)}</td>
                    <td>
                        <form method="POST" action="/models/download" style="display:inline">
                            <input type="hidden" name="repo" value="{html.escape(selected_repo)}">
                            <input type="hidden" name="filename" value="{html.escape(fname)}">
                            <button type="submit" class="btn btn-sm">Download</button>
                        </form>
                    </td>
                </tr>"""
            browse_html = f"""
            <h3>Files in {html.escape(selected_repo)}</h3>
            <table>
                <tr><th>Filename</th><th>Action</th></tr>
                {rows}
            </table>"""
        else:
            browse_html = f"<p>No GGUF files found in {html.escape(selected_repo)}.</p>"

    msg_html = ""
    if message:
        msg_html = f'<div class="alert success">{html.escape(message)}</div>'
    if error:
        msg_html = f'<div class="alert error">{html.escape(error)}</div>'

    llama_status = "running" if is_llama_running() else "stopped"
    status_class = "active" if is_llama_running() else "failed"

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Code Completion - Model Manager</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               background: #0d1117; color: #c9d1d9; padding: 20px; line-height: 1.5; }}
        h1 {{ color: #58a6ff; margin-bottom: 8px; }}
        h2 {{ color: #c9d1d9; margin: 24px 0 12px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
        h3 {{ color: #c9d1d9; margin: 16px 0 8px; }}
        a {{ color: #58a6ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        table {{ width: 100%; border-collapse: collapse; margin: 8px 0 16px; }}
        th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #21262d; }}
        th {{ color: #8b949e; font-weight: 600; }}
        .btn {{ background: #238636; color: #fff; border: none; padding: 6px 16px;
                border-radius: 6px; cursor: pointer; font-size: 14px; }}
        .btn:hover {{ background: #2ea043; }}
        .btn-sm {{ padding: 4px 12px; font-size: 13px; }}
        .btn-danger {{ background: #da3633; }}
        .btn-danger:hover {{ background: #f85149; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
                  font-size: 12px; font-weight: 600; }}
        .badge.active {{ background: #238636; color: #fff; }}
        .badge.downloading {{ background: #1f6feb; color: #fff; }}
        .badge.complete {{ background: #238636; color: #fff; }}
        .badge.failed {{ background: #da3633; color: #fff; }}
        input[type="text"] {{ background: #0d1117; border: 1px solid #30363d; color: #c9d1d9;
                              padding: 6px 12px; border-radius: 6px; font-size: 14px; width: 400px; }}
        .form-row {{ display: flex; gap: 8px; align-items: center; margin: 8px 0; flex-wrap: wrap; }}
        .alert {{ padding: 12px 16px; border-radius: 6px; margin: 12px 0; }}
        .alert.success {{ background: #0d2818; border: 1px solid #238636; }}
        .alert.error {{ background: #2d1117; border: 1px solid #da3633; color: #f85149; }}
        .status-bar {{ background: #161b22; padding: 12px 16px; border-radius: 6px;
                       margin-bottom: 16px; display: flex; gap: 16px; align-items: center; }}
        .hint {{ color: #8b949e; font-size: 13px; margin-top: 4px; }}
        .section {{ background: #161b22; padding: 16px; border-radius: 8px; margin: 12px 0; }}
    </style>
</head>
<body>
    <h1>Code Completion Server</h1>
    <div class="status-bar">
        <span>Server: <span class="badge {status_class}">{llama_status}</span></span>
        <span>Active model: <strong>{html.escape(active_model or 'none')}</strong></span>
        <span>API: <code>/v1/completions</code></span>
    </div>

    {msg_html}

    <h2>Downloaded Models</h2>
    <div class="section">
    {"<table><tr><th>Model</th><th>Size</th><th>Actions</th></tr>" + model_rows + "</table>" if models else "<p>No models downloaded yet.</p>"}
    </div>

    {"<h2>Downloads in Progress</h2><div class='section'><table><tr><th>File</th><th>Status</th></tr>" + download_rows + "</table></div>" if download_rows else ""}

    <h2>Download Models</h2>
    <div class="section">
        <h3>Search HuggingFace</h3>
        <form method="GET" action="/">
            <div class="form-row">
                <input type="text" name="search" placeholder="e.g. qwen2.5-coder gguf">
                <button type="submit" class="btn">Search</button>
            </div>
        </form>
        <p class="hint">Search for GGUF model repositories on HuggingFace.</p>

        {search_html}
        {browse_html}

        <h3>Direct Download</h3>
        <form method="POST" action="/models/download">
            <div class="form-row">
                <input type="text" name="repo" placeholder="owner/repo (e.g. bartowski/Qwen2.5-Coder-7B-Instruct-GGUF)">
            </div>
            <div class="form-row">
                <input type="text" name="filename" placeholder="filename.gguf">
                <button type="submit" class="btn">Download</button>
            </div>
        </form>
        <p class="hint">Enter a HuggingFace repo ID and GGUF filename to download directly.</p>
    </div>

    <h2>API Usage</h2>
    <div class="section">
        <p>This server exposes an OpenAI-compatible API. Configure your editor to point at:</p>
        <pre style="background:#0d1117;padding:12px;border-radius:6px;margin:8px 0;overflow-x:auto"><code>{{
  "tabAutocompleteModel": {{
    "title": "OpenHost Code Completion",
    "provider": "openai",
    "model": "current",
    "apiBase": "https://code-completion.&lt;your-zone&gt;/v1",
    "apiKey": "&lt;your-openhost-token&gt;"
  }}
}}</code></pre>
    </div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        # Health check
        if parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = "ok" if is_llama_running() else "no_model_loaded"
            self.wfile.write(json.dumps({"status": status}).encode())
            return

        # Proxy /v1/* to llama-server
        if parsed.path.startswith("/v1/"):
            if not is_llama_running():
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": {"message": "No model loaded. Visit the web UI to download and activate a model.", "type": "server_error"}
                }).encode())
                return
            proxy_headers = {}
            for key in ("content-type", "authorization", "accept"):
                val = self.headers.get(key)
                if val:
                    proxy_headers[key] = val
            status, resp_headers, resp_body = proxy_to_llama("GET", self.path, proxy_headers)
            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp_body)
            return

        # API: list models
        if parsed.path == "/api/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "models": list_models(),
                "active_model": get_active_model(),
                "downloads": dict(downloads),
            }).encode())
            return

        # Web UI
        if parsed.path == "/":
            search_results = None
            repo_files = None
            selected_repo = None

            if "search" in params:
                query = params["search"][0]
                search_results = search_huggingface(query)
            if "browse" in params:
                selected_repo = params["browse"][0]
                repo_files = list_repo_files(selected_repo)

            message = params.get("msg", [None])[0]
            error = params.get("err", [None])[0]

            body = render_page(
                models=list_models(),
                active_model=get_active_model() or "",
                download_status=dict(downloads),
                search_results=search_results,
                repo_files=repo_files,
                selected_repo=selected_repo,
                message=message,
                error=error,
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body.encode())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Proxy /v1/* to llama-server
        if parsed.path.startswith("/v1/"):
            if not is_llama_running():
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": {"message": "No model loaded. Visit the web UI to download and activate a model.", "type": "server_error"}
                }).encode())
                return
            proxy_headers = {}
            for key in ("content-type", "authorization", "accept"):
                val = self.headers.get(key)
                if val:
                    proxy_headers[key] = val
            status, resp_headers, resp_body = proxy_to_llama("POST", self.path, proxy_headers, body)
            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp_body)
            return

        # Parse form data
        content_type = self.headers.get("Content-Type", "")
        if "application/x-www-form-urlencoded" in content_type:
            form_data = parse_qs(body.decode())
        elif "application/json" in content_type:
            form_data = json.loads(body) if body else {}
        else:
            form_data = parse_qs(body.decode())

        def get_field(name):
            val = form_data.get(name)
            if isinstance(val, list):
                return val[0] if val else ""
            return val or ""

        def validate_filename(filename):
            """Reject path traversal and non-GGUF filenames."""
            if not filename or "/" in filename or "\\" in filename or ".." in filename:
                return False
            if not filename.endswith(".gguf"):
                return False
            return True

        # Download model
        if parsed.path == "/models/download":
            repo = get_field("repo").strip()
            filename = get_field("filename").strip()
            if not repo or not filename:
                self._redirect_with_error("Both repo and filename are required.")
                return
            if not validate_filename(filename):
                self._redirect_with_error("Invalid filename. Must be a .gguf file with no path separators.")
                return
            ok, err = download_model(repo, filename)
            if ok:
                self._redirect_with_message(f"Download started: {filename}")
            else:
                self._redirect_with_error(err or "Download failed")
            return

        # Activate model
        if parsed.path == "/models/activate":
            filename = get_field("filename").strip()
            if not filename:
                self._redirect_with_error("No model specified.")
                return
            if not validate_filename(filename):
                self._redirect_with_error("Invalid filename.")
                return
            model_path = MODELS_DIR / filename
            if not model_path.exists():
                self._redirect_with_error(f"Model not found: {filename}")
                return

            ok, err = start_llama(filename)
            if ok:
                self._redirect_with_message(f"Activated: {filename}")
            else:
                self._redirect_with_error(err or "Failed to start model")
            return

        # Delete model
        if parsed.path == "/models/delete":
            filename = get_field("filename").strip()
            if not filename:
                self._redirect_with_error("No model specified.")
                return
            if not validate_filename(filename):
                self._redirect_with_error("Invalid filename.")
                return
            active = get_active_model()
            if filename == active:
                with llama_lock:
                    stop_llama_locked()
                set_active_model(None)
            model_path = MODELS_DIR / filename
            if model_path.exists():
                model_path.unlink()
            with download_lock:
                downloads.pop(filename, None)
            self._redirect_with_message(f"Deleted: {filename}")
            return

        self.send_response(404)
        self.end_headers()

    def _redirect_with_message(self, msg):
        self.send_response(303)
        self.send_header("Location", f"/?msg={quote(msg)}")
        self.end_headers()

    def _redirect_with_error(self, err):
        self.send_response(303)
        self.send_header("Location", f"/?err={quote(err)}")
        self.end_headers()

    def log_message(self, format, *args):
        print(f"[HTTP] {args[0]}", flush=True)


def main():
    # Try to auto-start previously active model
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
