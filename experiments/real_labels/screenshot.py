"""Screenshot of the running browser labeller (headless Firefox), to check
that the page loads and an item renders.

Firefox's ``--screenshot`` fires at the page's load event, before the
labeller's scripts have fetched and drawn the item. So a throw-away local
page embeds the labeller in an iframe plus an image from an endpoint that
answers only after ``--delay`` seconds; the load event (and the screenshot)
then waits for it. Read-only: the labeller only receives GET requests.

    uv run python experiments/real_labels/screenshot.py --port 8765 --out shot.png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--delay", type=float, default=12.0)
    ap.add_argument("--size", default="1700,1000")
    ap.add_argument("--firefox", default=shutil.which("firefox") or str(Path.home() / ".local/bin/firefox"))
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split(","))
    page = (f'<!doctype html><body style="margin:0"><iframe src="http://localhost:{args.port}/" '
            f'style="border:0;width:{w}px;height:{h}px"></iframe><img src="/wait" style="display:none">'
            "</body>").encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path == "/wait":
                time.sleep(args.delay)
                body, kind = b"GIF89a\x01\x00\x01\x00\x00\x00\x00;", "image/gif"
            else:
                body, kind = page, "text/html"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [args.firefox, "--headless", "--no-remote", "--profile", profile,
             f"--window-size={w},{h}", "--screenshot", str(args.out.resolve()),
             f"http://127.0.0.1:{srv.server_address[1]}/"],
            check=True, timeout=120 + args.delay, capture_output=True,
        )
    srv.shutdown()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
