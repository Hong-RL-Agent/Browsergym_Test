from __future__ import annotations

import argparse
from urllib.parse import urlparse

import browsergym.core  # noqa: F401
import gymnasium as gym


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:9220")
    args = parser.parse_args()

    print(f"[openended-test] requested base_url: {args.base_url}")
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": args.base_url},
        wait_for_user_message=False,
    )

    try:
        obs, info = env.reset()
        print("[openended-test] obs url:", obs.get("url"))
        print("[openended-test] open_pages_urls:", obs.get("open_pages_urls"))
        print("[openended-test] open_pages_titles:", obs.get("open_pages_titles"))
        _warn_if_url_mismatch(args.base_url, obs.get("url") or _first_url(obs.get("open_pages_urls")))

        print("OBS KEYS:", obs.keys())
        print("INFO:", info)

        for key, value in obs.items():
            print("\n" + "=" * 80)
            print("KEY:", key)
            print("TYPE:", type(value))

            if isinstance(value, str):
                print("VALUE:", value[:1000])
            elif isinstance(value, list):
                print("LENGTH:", len(value))
                print("SAMPLE:", value[:3])
            elif isinstance(value, dict):
                print("DICT KEYS:", list(value.keys())[:30])
                print("SAMPLE:", str(value)[:1000])
            else:
                print("VALUE:", str(value)[:1000])
        return 0
    finally:
        env.close()


def _first_url(value: object) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return ""


def _warn_if_url_mismatch(requested_url: str, observed_url: object) -> None:
    requested = urlparse(requested_url)
    observed = urlparse(str(observed_url or ""))
    if requested.netloc and observed.netloc and requested.netloc != observed.netloc:
        print(f"WARNING: requested base_url is {requested_url} but BrowserGym opened {observed_url}")


if __name__ == "__main__":
    raise SystemExit(main())
