"""Browser labeller for fish keypoints in video frames (local server, standard library only).

    python experiments/video2022/pose_labeller/server.py SESSION_DIR [--port 8766] [--labeller weygoldt]

then open http://localhost:8766/ (from a laptop: ``ssh -N -L 8766:localhost:8766 polarbear``).

A session directory (written by ``b2_session.py``) holds ``items.json`` (the
queue: item id, image file, image size, optional pre-filled fish), ``images/``
and, once labelling starts, ``labels.jsonl`` (one line per save, the latest
line of an item wins; see ``pose_labels.py``). Nothing else in the session
directory (e.g. ``truth.csv`` with the recording and frame of each item) is
ever served. Every save is written at once; stopping the server or closing
the tab loses nothing, and restarting resumes where the labels end.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pose_labels as pl  # noqa: E402

HTML = Path(__file__).resolve().parent / "labeller.html"


LEGACY_SKELETON = {"nodes": ["snout", "spine1", "tail"], "edges": [["snout", "spine1"], ["spine1", "tail"]],
                   "note": "centre = spine1: just behind the head, at the pectoral fins"}
DEFAULT_COLORS = {"snout": "#ff4d4d", "spine1": "#ffd24d", "spine2": "#c7e9b4", "spine3": "#7fcdbb",
                  "tail": "#4dc3ff", "left_fin": "#f781bf", "right_fin": "#a65628"}


class Session:
    """A labelling session directory.

    ``session.json`` (optional) names the skeleton: ``nodes`` (click order),
    ``edges`` (lines drawn), ``colors`` and a ``note`` for the help panel.
    Without it the session is a 3-point one (snout, spine1, tail), as b2_v1.
    """

    def __init__(self, root: Path, labeller: str) -> None:
        self.root = Path(root)
        self.items = json.loads((self.root / "items.json").read_text())
        self.ids = {it["item_id"] for it in self.items}
        self.labels_path = self.root / "labels.jsonl"
        self.labeller = labeller
        self.lock = threading.Lock()
        cfg = self.root / "session.json"
        self.skeleton = json.loads(cfg.read_text()) if cfg.exists() else dict(LEGACY_SKELETON)
        self.skeleton.setdefault("colors", {n: DEFAULT_COLORS.get(n, "#ffffff") for n in self.skeleton["nodes"]})
        unknown = set(self.skeleton["nodes"]) - set(pl.SKELETON)
        if unknown:
            raise ValueError(f"{cfg}: unknown nodes {sorted(unknown)}")

    def state(self) -> dict:
        labels = {k: v.to_json() for k, v in pl.load_labels(self.labels_path).items()}
        public = [{"item_id": it["item_id"], "image": it["image"], "width": it["width"], "height": it["height"],
                   "prefill": [pl.Fish.from_dict(f).to_json() for f in it.get("prefill", [])]}
                  for it in self.items]
        return {"items": public, "labels": labels, "labeller": self.labeller, "skeleton": self.skeleton}

    def save(self, payload: dict) -> dict:
        if payload.get("item_id") not in self.ids:
            raise ValueError(f"unknown item {payload.get('item_id')!r}")
        payload = {**payload, "labeller": self.labeller,
                   "labelled_at": datetime.now(UTC).isoformat(timespec="milliseconds")}
        label = pl.FrameLabel.from_json(payload)  # validates
        with self.lock:
            pl.append_label(self.labels_path, label)
        return label.to_json()


def make_handler(session: Session) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: object, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/session":
                self._json(session.state())
            elif self.path.startswith("/images/"):
                name = Path(self.path).name
                path = session.root / "images" / name
                if path.parent != session.root / "images" or not path.is_file():
                    self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                else:
                    self._send(200, path.read_bytes(), "image/jpeg")
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/api/label":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                self._json({"ok": True, "label": session.save(json.loads(self.rfile.read(n)))})
            except (ValueError, KeyError, TypeError) as e:
                self._json({"ok": False, "error": str(e)}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(root: Path, port: int, labeller: str) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Session(root, labeller)))
    return httpd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--labeller", default="weygoldt")
    args = ap.parse_args()
    httpd = serve(args.session, args.port, args.labeller)
    print(f"labeller on http://localhost:{args.port}/ (session {args.session})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
