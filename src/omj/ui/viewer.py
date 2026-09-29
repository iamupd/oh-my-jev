"""`omj bench --view`: show a finished run on the Reports page.

If an `omj ui` server already answers on the port, the browser is pointed at it. Otherwise a
reports-only server (no models loaded) is started in the foreground until Ctrl+C.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from urllib.parse import quote

import httpx

HOST = "127.0.0.1"
DEFAULT_PORT = 8800


def is_omj_ui(port: int, host: str = HOST, timeout: float = 1.0) -> bool:
    """True when an omj ui server answers on host:port."""
    try:
        response = httpx.get(f"http://{host}:{port}/ui/api/health", timeout=timeout)
        return response.status_code == 200 and response.json().get("app") == "omj-ui"
    except (httpx.HTTPError, ValueError):
        return False


def port_in_use(port: int, host: str = HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def free_port(host: str = HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def report_url(port: int, run_id: str | None, host: str = HOST) -> str:
    query = f"?run={quote(run_id)}" if run_id else ""
    return f"http://{host}:{port}/reports{query}"


def _serve_reports(port: int, on_ready: Callable[[], None]) -> None:
    import uvicorn

    from omj.ui.app import create_ui_app

    def open_when_up() -> None:
        for _ in range(100):
            if is_omj_ui(port, timeout=0.3):
                on_ready()
                return
            time.sleep(0.1)

    threading.Thread(target=open_when_up, daemon=True).start()
    uvicorn.run(create_ui_app([]), host=HOST, port=port, log_level="warning")


def view_run(
    run_id: str | None,
    *,
    port: int = DEFAULT_PORT,
    opener: Callable[[str], object] = webbrowser.open,
    serve: Callable[[int, Callable[[], None]], None] = _serve_reports,
    out=None,
) -> str:
    """Open the run in a browser; returns "reused" or "served" (after the server stops)."""
    out = out if out is not None else sys.stderr
    if is_omj_ui(port):
        url = report_url(port, run_id)
        print(f"Opening {url} (omj ui is already running)", file=out)
        opener(url)
        return "reused"
    if port_in_use(port):
        port = free_port()
    url = report_url(port, run_id)
    print(f"Serving reports on {url} (Ctrl+C to stop)", file=out)
    try:
        serve(port, lambda: opener(url))
    except KeyboardInterrupt:
        pass
    return "served"
