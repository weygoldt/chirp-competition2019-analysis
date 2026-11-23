"""Browser tool to review the BORIS chase onsets and offsets of the 2019/2020 competitions (standard library + ffmpeg).

    python paper/chase_review/server.py SESSION_DIR [--port 8769] [--reviewer weygoldt]

then open http://localhost:8769/ (from a laptop: ``ssh -N -L 8769:localhost:8769 polarbear``).
``serve.sh NAME`` starts it as a systemd user service.

A session directory (``build_session.py``) holds ``session.json``. The server cuts the
video of each chase's window on request from the video BORIS was scored on, caches it in
``clips/`` (safe to delete) and serves it with HTTP range requests, so the browser can
step frame by frame. Each save appends one line to ``revisions.jsonl``; the latest line
of a chase wins. The BORIS files, videos and LED files are never written.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import queue
import re
import subprocess
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = {"": "index.html", "index.html": "index.html", "timemap.js": "timemap.js"}
MAX_CLIP_FRAMES = 25 * 600  # 10 min of video per clip at most
CLIP_RE = re.compile(r"clip/([0-9A-Za-z_-]+)/(\d+)-(\d+)\.mp4")


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


def to_rec_s(led_map: dict, frame: float) -> float:
    """Recording time (s) of a video frame: ``led_alignment.LedAlignment.to_grid`` on the stored segments."""
    k = bisect.bisect_right(led_map["bounds"], frame)
    return led_map["slope"][k] * frame + led_map["icpt"][k]


def clip_command(video: Path, frame0: int, n: int, fps: float, delay: int, out: Path) -> list[str]:
    """ffmpeg call for frames frame0 .. frame0 + n - 1 of ``video`` (numbering of LED_frames.npy).

    Seeks to (frame0 + delay) / fps, the timestamp ffmpeg gives frame frame0 (see
    build_session.probe_video), and re-encodes without B-frames and with a keyframe every
    12 frames, so the browser seeks fast and clip frame i sits at i / fps.
    """
    return ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{(frame0 + delay) / fps:.6f}", "-i", str(video),
            "-frames:v", str(n), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "12",
            "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(out)]


def count_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().strip(","))


def make_clip(video: Path, frame0: int, n: int, fps: float, delay: int, out: Path) -> Path:
    """Cut the clip into ``out`` (atomically); raise if it does not hold exactly n frames."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    subprocess.run(clip_command(video, frame0, n, fps, delay, tmp), check=True, capture_output=True)
    got = count_frames(tmp)
    if got != n:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"clip {out.name}: {got} frames, expected {n}")
    tmp.replace(out)
    return out


def check_revision(rev: dict, chases: dict[str, dict], recordings: dict[str, dict]) -> dict:
    """A revision as saved: chase id, revised onset / offset (video frames), flags and note.

    ``not_a_chase``: the BORIS event is no chase (then the frames may be null).
    ``onset_unsure`` / ``offset_unsure``: the reviewer could not place that end well.
    ``blind``: the BORIS markers were hidden the whole time this chase was open.
    ``chirps_hidden``: the chirps were hidden the whole time (chirps come from the electric
    recording and are only shown, never edited).
    """
    cid = rev.get("chase_id")
    if cid not in chases:
        raise ValueError(f"unknown chase {cid!r}")
    chase = chases[cid]
    info = recordings[chase["recording"]]
    not_a_chase = bool(rev.get("not_a_chase", False))
    on, off = rev.get("onset_frame"), rev.get("offset_frame")
    if not_a_chase and (on is None or off is None):
        on = off = None
    else:
        if on is None or off is None:
            raise ValueError("mark both the onset and the offset (or tick 'not a chase')")
        on, off = int(on), int(off)
        if not 0 <= on < off < info["n_frames"]:
            raise ValueError(f"onset frame {on} must be >= 0 and before offset frame {off} (video has "
                             f"{info['n_frames']} frames)")
    out = {"chase_id": cid, "recording": chase["recording"], "k": chase["k"],
           "onset_frame": on, "offset_frame": off, "not_a_chase": not_a_chase,
           "onset_unsure": bool(rev.get("onset_unsure", False)),
           "offset_unsure": bool(rev.get("offset_unsure", False)),
           "blind": bool(rev.get("blind", False)), "chirps_hidden": bool(rev.get("chirps_hidden", False)),
           "note": str(rev.get("note", ""))[:1000],
           "boris_onset_frame": chase["onset_frame"], "boris_offset_frame": chase["offset_frame"],
           "boris_onset_rec_s": chase["onset_rec_s"], "boris_offset_rec_s": chase["offset_rec_s"]}
    if on is not None:
        out["onset_rec_s"] = to_rec_s(info["led_map"], on)
        out["offset_rec_s"] = to_rec_s(info["led_map"], off)
        out["d_onset_s"] = out["onset_rec_s"] - chase["onset_rec_s"]
        out["d_offset_s"] = out["offset_rec_s"] - chase["offset_rec_s"]
    win = rev.get("window")
    if isinstance(win, dict):
        out["window"] = {"frame0": int(win["frame0"]), "n": int(win["n"])}
    return out


def latest(path: Path) -> dict[str, dict]:
    """The latest saved revision of each chase."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rev = json.loads(line)
                out[rev["chase_id"]] = rev
    return out


class Session:
    def __init__(self, root: Path, reviewer: str, clips: Path | None = None) -> None:
        self.root = Path(root)
        self.raw = (self.root / "session.json").read_bytes()
        self.data = json.loads(self.raw)
        self.data_root = Path(self.data["meta"]["data_root"])
        self.recordings = self.data["recordings"]
        self.chases = {c["id"]: c for c in self.data["chases"]}
        self.revisions_path = self.root / "revisions.jsonl"
        self.clips = Path(clips) if clips else self.root / "clips"
        self.reviewer = reviewer
        self.lock = threading.Lock()
        self.clip_locks: dict[Path, threading.Lock] = {}
        self.prefetch_q: queue.Queue[tuple[str, int, int]] = queue.Queue()
        threading.Thread(target=self._prefetch_worker, daemon=True).start()

    def clip_path(self, rec: str, frame0: int, n: int) -> Path:
        if rec not in self.recordings:
            raise KeyError(rec)
        info = self.recordings[rec]
        if not (0 < n <= MAX_CLIP_FRAMES and frame0 >= 0 and frame0 + n <= info["n_frames"]):
            raise ValueError(f"frames {frame0}+{n} outside the video ({info['n_frames']} frames, "
                             f"at most {MAX_CLIP_FRAMES} per clip)")
        return self.clips / rec / f"{frame0:07d}_{n:05d}.mp4"

    def clip(self, rec: str, frame0: int, n: int) -> Path:
        out = self.clip_path(rec, frame0, n)
        with self.lock:
            lock = self.clip_locks.setdefault(out, threading.Lock())
        with lock:
            if not out.exists():
                info = self.recordings[rec]
                make_clip(self.data_root / rec / info["video"], frame0, n, info["fps"], info["delay_frames"], out)
        return out

    def prefetch(self, rec: str, frame0: int, n: int) -> None:
        self.clip_path(rec, frame0, n)  # validate now, cut later
        self.prefetch_q.put((rec, frame0, n))

    def _prefetch_worker(self) -> None:
        while True:
            rec, frame0, n = self.prefetch_q.get()
            try:
                self.clip(rec, frame0, n)
            except Exception as e:  # a failed prefetch is retried on request
                print(f"prefetch {rec} {frame0}+{n} failed: {e}", flush=True)

    def save(self, rev: dict) -> dict:
        rec = check_revision(rev, self.chases, self.recordings) | {
            "reviewer": self.reviewer, "saved_at": datetime.now(UTC).isoformat(timespec="seconds")}
        with self.lock, open(self.revisions_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec


def make_handler(session: Session) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def send_body(self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_body(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self) -> None:
            path = self.path.split("?")[0].lstrip("/")
            if path in STATIC:
                f = HERE / STATIC[path]
                ctype = "text/javascript" if f.suffix == ".js" else "text/html; charset=utf-8"
                self.send_body(f.read_bytes(), ctype)
            elif path == "api/session":
                self.send_body(session.raw, "application/json")
            elif path == "api/revisions":
                self.send_json(latest(session.revisions_path))
            elif m := CLIP_RE.fullmatch(path):
                try:
                    f = session.clip(m.group(1), int(m.group(2)), int(m.group(3)))
                except (KeyError, ValueError) as e:
                    self.send_json({"error": str(e)}, HTTPStatus.NOT_FOUND)
                    return
                except (RuntimeError, subprocess.CalledProcessError) as e:
                    self.send_json({"error": f"clip failed: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                self.send_video(f)
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
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):  # the browser cancelled
                    self.wfile.write(fh.read(last - first + 1))

        def do_POST(self) -> None:
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if self.path == "/api/save":
                    self.send_json(session.save(body))
                elif self.path == "/api/prefetch":
                    session.prefetch(body["recording"], int(body["frame0"]), int(body["n"]))
                    self.send_json({"queued": True})
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError, TypeError) as e:
                self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", type=Path, help="session directory (build_session.py)")
    ap.add_argument("--port", type=int, default=8769)
    ap.add_argument("--reviewer", default="weygoldt")
    ap.add_argument("--clips", type=Path, help="clip cache (default SESSION/clips)")
    args = ap.parse_args()
    session = Session(args.session, args.reviewer, args.clips)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    n_done = len(latest(session.revisions_path))
    print(f"chase review: http://localhost:{args.port}/  ({len(session.chases)} chases, {n_done} reviewed)",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
