"""Start the validated Run6 site100-102 servers without changing site code."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"C:\workspace\RL_Errorsite_fullstack_run4")
LOG = ROOT / "artifacts/training_v3/run6_confirmation_aware_dqn/seed42_full/server_logs"
NPM = Path(r"C:\Program Files\nodejs\npm.cmd")


def listening(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start(command: list[str], cwd: Path, log_name: str) -> dict[str, object]:
    LOG.mkdir(parents=True, exist_ok=True)
    log = (LOG / log_name).open("a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=log,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    return {"pid": proc.pid, "cwd": str(cwd), "command": command, "log": str(LOG / log_name)}


def main() -> None:
    sites = []
    for directory in sorted(SOURCE.iterdir(), key=lambda p: p.name):
        if not directory.is_dir() or not directory.name.startswith(("site100_", "site101_", "site102_")):
            continue
        number = int(directory.name[4:7])
        frontend_port = 5200 + number - 100
        backend_port = 9599 + number - 100
        site = {"site": f"SITE{number}", "path": str(directory), "frontend_port": frontend_port, "backend_port": backend_port, "processes": []}
        if not listening(backend_port):
            site["processes"].append(start([str(NPM), "start"], directory / "backend", f"SITE{number}_backend.runtime.log"))
        if not listening(frontend_port):
            site["processes"].append(start([str(NPM), "run", "dev", "--", "--host", "127.0.0.1"], directory / "frontend", f"SITE{number}_frontend.runtime.log"))
        sites.append(site)
    time.sleep(8)
    for site in sites:
        site["backend_listening"] = listening(int(site["backend_port"]))
        site["frontend_listening"] = listening(int(site["frontend_port"]))
    (LOG / "server_manifest.json").write_text(json.dumps({"sites": sites}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sites": sites}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
