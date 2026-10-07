"""Read-only Run6 BrowserGym/Playwright infrastructure diagnostics."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path


OUT = Path(__file__).resolve().parents[2] / "artifacts/training_v3/run6_infrastructure_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)


def env_contract() -> dict[str, str | None]:
    keys = ("USERPROFILE", "HOME", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP", "PLAYWRIGHT_BROWSERS_PATH")
    return {key: os.environ.get(key) for key in keys}


def standalone() -> dict:
    result = {"pass": False, "primary_exception": None, "cleanup_exception": None}
    pw = browser = context = page = None
    try:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        result["playwright_type"] = type(pw).__name__
        result["browser_executable"] = pw.chromium.executable_path
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.set_content("<html><body>ok</body></html>")
        result.update({"pass": True, "url": page.url, "title": page.title()})
    except Exception as exc:
        result["primary_exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        for name, obj in (("page", page), ("context", context), ("browser", browser)):
            if obj is not None:
                try:
                    obj.close()
                except Exception as exc:
                    result["cleanup_exception"] = {"component": name, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if pw is not None:
            try:
                pw.stop()
            except Exception as exc:
                result["cleanup_exception"] = {"component": "playwright", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    return result


def browsergym_one(site: dict) -> dict:
    result = {"pass": False, "primary_exception": None, "cleanup_exception": None}
    env = None
    server = None
    try:
        from dqn_error_discovery.v3.run4_fullstack_training import SiteServer
        server = SiteServer(site)
        server.__enter__()
        from envs.browsergym_jaws_env import BrowserGymJAWSEnv
        env = BrowserGymJAWSEnv(site_id=str(site["site_id"]), base_url=str(site["url"]), max_steps=10, max_candidates=32, headless=True)
        raw, info = env.reset()
        result.update({"pass": True, "observation_keys": sorted(raw.keys()) if isinstance(raw, dict) else [], "info_keys": sorted(info.keys()) if isinstance(info, dict) else []})
    except Exception as exc:
        result["primary_exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result["cleanup_exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if server is not None:
            try:
                server.__exit__(None, None, None)
            except Exception as exc:
                result["cleanup_exception"] = {"component": "site_server", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    return result


def short_train_smoke(site: dict) -> dict:
    """Exercise the existing Run6 runner for five episodes without changing RL code."""
    try:
        from argparse import Namespace
        from dqn_error_discovery.v3.run5_context_aware_dqn_training import train_seed
        args = Namespace(
            episodes_per_site=40, dqn_sessions_per_site=2, max_local_steps=5,
            max_steps=10, browser_timeout_ms=60000, epsilon=0.1,
            batch_size=32, target_update_interval=25, disk_min_free_bytes=0,
            optimizer_smoke=True, max_smoke_episodes=40,
        )
        summary = train_seed(42, [site], args)
        return {
            "pass": int(summary.get("episodes", 0)) >= 5 and int(summary.get("replay_sampling", {}).get("batches", 0)) >= 5 and int(summary.get("optimizer_updates", 0)) >= 5,
            "summary": summary,
        }
    except Exception as exc:
        return {"pass": False, "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}


def site006_three_episodes() -> dict:
    rows = []
    process = stdout = stderr = None
    try:
        from dqn_error_discovery.v3.confirmation_forced_reproduction_diagnostic import _start_site_server
        from envs.browsergym_jaws_env import BrowserGymJAWSEnv
        process, stdout, stderr = _start_site_server()
        for episode in range(3):
            env = None
            try:
                env = BrowserGymJAWSEnv(site_id="site006", base_url="http://localhost:9115/", max_steps=5, max_candidates=32, headless=True)
                raw, info = env.reset()
                rows.append({"episode": episode + 1, "reset": True, "observation": bool(raw), "info": bool(info)})
            except Exception as exc:
                rows.append({"episode": episode + 1, "reset": False, "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}})
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
    except Exception as exc:
        return {"pass": False, "rows": rows, "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
    finally:
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        for handle in (stdout, stderr):
            if handle is not None:
                handle.close()
    return {"pass": len(rows) == 3 and all(row.get("reset") for row in rows), "rows": rows}


def path_probe(path: str) -> dict:
    import os
    result = {"path": path, "exists": False, "lexists": False}
    try:
        st = os.lstat(path)
        result.update({"exists": os.path.exists(path), "lexists": os.path.lexists(path), "is_directory": os.path.isdir(path), "is_symlink": os.path.islink(path), "mode": oct(st.st_mode)})
    except Exception as exc:
        result["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    try:
        node = subprocess.run(["node", "-e", "const fs=require('fs'); const p=process.argv[1]; try { const s=fs.lstatSync(p); console.log(JSON.stringify({ok:true,isDirectory:s.isDirectory(),isSymbolicLink:s.isSymbolicLink(),mode:s.mode.toString(8)})); } catch(e) { console.log(JSON.stringify({ok:false,code:e.code,message:e.message,stack:e.stack})); }", path], capture_output=True, text=True, timeout=15)
        result["node_lstat"] = node.stdout.strip() or node.stderr.strip()
    except Exception as exc:
        result["node_probe_exception"] = str(exc)
    return result


def main() -> None:
    registry = Path(os.environ.get("RAWD_RUN6_REGISTRY", str(Path(__file__).resolve().parents[2] / "artifacts/training_v3/run6_confirmation_aware_dqn/seed42_fresh/site100_plus_registry.json")))
    payload = {"python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "env": env_contract(), "path_probe": path_probe(r"C:\Users\META33"), "standalone": standalone()}
    if registry.exists():
        sites = json.loads(registry.read_text(encoding="utf-8")).get("sites", [])
        if sites:
            payload["browsergym_site"] = {"site_id": sites[0].get("site_id"), "url": sites[0].get("url")}
            payload["browsergym_one"] = browsergym_one(sites[0])
            if os.environ.get("RAWD_RUN6_REPEAT") == "1":
                payload["standalone_repeated"] = [standalone() for _ in range(5)]
                payload["browsergym_repeated"] = [browsergym_one(sites[0]) for _ in range(5)]
                payload["site006_three_episodes"] = site006_three_episodes()
            if os.environ.get("RAWD_RUN6_SHORT_SMOKE") == "1":
                payload["run6_short_train_smoke"] = short_train_smoke(sites[0])
    (OUT / "diagnostic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
