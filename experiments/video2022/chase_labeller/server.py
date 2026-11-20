"""Browser labeller for chase onsets and offsets in video clips (local server, standard library only).

    python experiments/video2022/chase_labeller/server.py SESSION_DIR [--port 8768] [--labeller weygoldt]

then open http://localhost:8768/ (from a laptop: ``ssh -N -L 8768:localhost:8768 polarbear``).

A session directory (``c6_chase_label_session.py``) holds ``clips.json`` (clip id,
the raw and the marked video file, frame rate) and ``clips/``. Only those are
served (``truth.csv`` with the detector's times is not). Each save appends one line
to ``labels.jsonl``; the latest line of a clip wins. Videos are served with HTTP
range requests, so the browser can seek frame by frame.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HTML = Path(__file__).resolve().parent / "index.html"


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """(first, last) byte of a ``Range: bytes=a-b`` header, or None for the whole file."""
    if not header:
        return None
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return None
    if m.group(1) == "":  # suffix range: the last n bytes
        n = int(m.group(2))
        return max(0, size - n), size - 1
    first = int(m.group(1))
    last = int(m.group(2)) if m.group(2) else size - 1
    if first >= size:
        raise ValueError("range starts after the end")
    return first, min(last, size - 1)


def check_label(lab: dict, clip_ids: set[str]) -> dict:
    """A label as saved: clip id, chases (onset / offset in clip seconds), flags, note."""
    if lab.get("clip_id") not in clip_ids:
        raise ValueError(f"unknown clip {lab.get('clip_id')!r}")
    chases = []
    for c in lab.get("chases", []):
        on, off = float(c["onset_s"]), float(c["offset_s"])
        if not 0 <= on < off:
            raise ValueError(f"onset {on} must be >= 0 and before offset {off}")
        chases.append({"onset_s": on, "offset_s": off, "continues": bool(c.get("continues", False))})
    if lab.get("no_chase") and chases:
        raise ValueError("'no chase' together with marked chases")
    return {"clip_id": lab["clip_id"], "chases": sorted(chases, key=lambda c: c["onset_s"]),
            "no_chase": bool(lab.get("no_chase", False)), "unsure": bool(lab.get("unsure", False)),
            "note": str(lab.get("note", ""))[:500]}


def latest(path: Path) -> dict[str, dict]:
    """The latest saved label of each clip."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                lab = json.loads(line)
                out[lab["clip_id"]] = lab
    return out


class Session:
    def __init__(self, root: Path, labeller: str) -> None:
        self.root = Path(root)
        self.clips = json.loads((self.root / "clips.json").read_text())
        self.ids = {c["clip_id"] for c in self.clips}
        self.files = {c[k] for c in self.clips for k in ("video", "video_marked")}
        self.labels_path = self.root / "labels.jsonl"
        self.labeller = labeller
        self.lock = threading.Lock()

    def save(self, lab: dict) -> dict:
        rec = check_label(lab, self.ids) | {"labeller": self.labeller, "saved_at": datetime.now(UTC).isoformat()}
        with self.lock, open(self.labels_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec


def make_handler(session: Session) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def send_json(self, obj: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?")[0].lstrip("/")
            if path in ("", "index.html"):
                body = HTML.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "api/clips":
                self.send_json(session.clips)
            elif path == "api/labels":
                self.send_json(latest(session.labels_path))
            elif path in session.files:
                self.send_video(session.root / path)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def send_video(self, f: Path) -> None:
            size = f.stat().st_size
            try:
                rng = parse_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            first, last = rng if rng else (0, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT if rng else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(last - first + 1))
            if rng:
                self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
            self.end_headers()
            with open(f, "rb") as fh:
                fh.seek(first)
                self.wfile.write(fh.read(last - first + 1))

        def do_POST(self) -> None:
            if self.path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                lab = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self.send_json(session.save(lab))
            except (ValueError, KeyError, TypeError) as e:
                self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--labeller", default="weygoldt")
    args = ap.parse_args()
    session = Session(Path(args.session), args.labeller)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    print(f"chase labeller: http://localhost:{args.port}/  ({len(session.clips)} clips)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
