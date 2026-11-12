"""Stop guarded chirp_v2 jobs before the machine runs out of RAM or VRAM.

Runs until killed. Every few seconds it reads MemAvailable from
/proc/meminfo and the GPU's used memory from nvidia-smi. When available RAM
falls below --min-ram-gb, or used VRAM rises above --max-vram-gb (whoever
uses it), it stops every systemd user scope named ``chirpv2-*`` (started by
``scripts/guarded``), newest first, one per check, and logs what it did.
Jobs outside those scopes are never touched.

    python scripts/oom_watchdog.py --log /mnt/data1/chirp_v2/logs/watchdog.log
"""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


#: Name fragments of exploratory jobs, stopped before training runs.
LOW_PRIORITY = ("params", "explore", "showcase", "smoke", "selftest", "probe", "prof")


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("inf")


def vram_used_gb() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        return max(float(v) for v in out.split()) / 1024
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


def guarded_scopes() -> list[str]:
    out = subprocess.run(
        ["systemctl", "--user", "list-units", "--type=scope", "--state=running",
         "--no-legend", "--plain", "chirpv2-*"],
        capture_output=True, text=True, check=False,
    ).stdout
    units = [line.split()[0] for line in out.splitlines() if line.strip()]

    def stamp(unit: str) -> str:
        # scripts/guarded names end in ...-YYYYMMDD-HHMMSS-PID
        return "-".join(unit.removesuffix(".scope").split("-")[-3:-1])

    def low_priority(unit: str) -> bool:
        return any(tag in unit for tag in LOW_PRIORITY)

    # Exploratory jobs first, then everything else; newest first within each.
    newest_first = sorted(units, key=stamp, reverse=True)
    return sorted(newest_first, key=lambda u: not low_priority(u))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ram-gb", type=float, default=6.0)
    ap.add_argument("--max-vram-gb", type=float, default=15.0)
    ap.add_argument("--interval-s", type=float, default=3.0)
    ap.add_argument("--log", type=Path, required=True)
    args = ap.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with args.log.open("a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")

    log(f"start min_ram_gb={args.min_ram_gb} max_vram_gb={args.max_vram_gb}")
    last_ok = 0.0
    while True:
        ram, vram = mem_available_gb(), vram_used_gb()
        if ram < args.min_ram_gb or vram > args.max_vram_gb:
            scopes = guarded_scopes()
            if scopes:
                victim = scopes[0]
                log(f"LOW ram_avail={ram:.1f}GB vram_used={vram:.1f}GB -> stopping {victim}")
                subprocess.run(["systemctl", "--user", "stop", victim], check=False)
            else:
                log(f"LOW ram_avail={ram:.1f}GB vram_used={vram:.1f}GB, no guarded job to stop")
            time.sleep(10)
            continue
        if time.monotonic() - last_ok > 600:
            log(f"ok ram_avail={ram:.1f}GB vram_used={vram:.1f}GB jobs={len(guarded_scopes())}")
            last_ok = time.monotonic()
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
