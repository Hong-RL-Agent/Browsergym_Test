from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import urlparse


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies
from services.known_bug_matcher import load_known_bugs, match_anomalies_to_known_bugs
from services.site_profile_service import build_site_profile, load_training_site_config, validate_site_identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-id", default="manual-test")
    parser.add_argument("--base-url", default="http://localhost:9220")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument(
        "--debug-action",
        choices=["noop", "inspect_layout", "inspect_dom", "inspect_network", "inspect_console", "inspect_cart"],
        default="noop",
    )
    parser.add_argument("--exploration-profile", default="openended_commerce")
    parser.add_argument("--config", default="configs/training_sites.json")
    args = parser.parse_args()

    site_config = load_training_site_config(args.site_id, args.config)
    known_bugs = load_known_bugs(args.site_id)
    site_profile = build_site_profile(
        args.site_id,
        known_bugs,
        exploration_profile=site_config.get("exploration_profile") or (args.exploration_profile if not known_bugs else None),
    )

    env = BrowserGymJAWSEnv(
        site_id=args.site_id,
        base_url=args.base_url,
        max_steps=args.max_steps,
        max_candidates=args.max_candidates,
        site_profile=site_profile,
        requires_login=bool(site_config.get("requires_login")),
        login_config=site_config.get("login") if isinstance(site_config.get("login"), dict) else None,
    )
    try:
        obs, info = env.reset()
        print("reset succeeded")
        print("[jaws-env-test] requested site_id:", args.site_id)
        print("[jaws-env-test] requested base_url:", args.base_url)
        print("[jaws-env-test] login_required:", bool(info.get("login_required")))
        print("[jaws-env-test] login_attempted:", bool(info.get("login_attempted")))
        print("[jaws-env-test] login_success:", bool(info.get("login_success")))
        print("[jaws-env-test] post_login_url:", info.get("post_login_url"))
        if info.get("login_required") and not info.get("login_success"):
            print("WARNING: site9800 login failed; still observing login page.")
        print("[jaws-env-test] observed url:", obs["page_state"].get("url"))
        print("[jaws-env-test] observed title:", obs["page_state"].get("title"))
        print("[jaws-env-test] candidate_count:", len(obs.get("candidate_elements", [])))
        print("[jaws-env-test] page_text_sample:", _safe(str(obs.get("page_state", {}).get("page_text_sample", ""))[:1000]))
        identity = validate_site_identity(args.site_id, obs)
        if not identity["data_bug_ids_found"]:
            identity["site_identity_match"] = "unknown"
        print("[jaws-env-test] data_bug_ids_found:", identity["data_bug_ids_found"])
        print("[jaws-env-test] expected_bug_id_prefix:", identity["expected_bug_id_prefix"])
        print("[jaws-env-test] site_identity_match:", identity["site_identity_match"])
        for warning in identity["identity_warnings"]:
            print(warning)
        if known_bugs and bool(info.get("login_success")) and not identity["data_bug_ids_found"]:
            print("WARNING: bug_catalog loaded and login succeeded, but no data-bug-id was found. Check site9800 UI data-bug-id placement.")
        print("[jaws-env-test] has_empty_state_text:", obs.get("page_state", {}).get("has_empty_state_text"))
        print("[jaws-env-test] has_chart_like_element:", obs.get("page_state", {}).get("has_chart_like_element"))
        print("[jaws-env-test] site9800_bug_catalog_loaded:", bool(known_bugs) if args.site_id == "site9800" else "n/a")
        print("[jaws-env-test] bug_catalog_bug_ids:", [bug.get("bug_id") or bug.get("id") for bug in known_bugs])
        print("[jaws-env-test] dom_attributes_summary:", obs.get("page_state", {}).get("dom_attributes_summary", {}))
        _print_candidate_debug(obs)
        _warn_if_url_mismatch(args.base_url, obs["page_state"].get("url"))
        print("J.A.W.S observation keys:", list(obs.keys()))
        print("candidate_count:", len(obs.get("candidate_elements", [])))
        _print_top_candidates(obs)

        obs_vector = env.observation_encoder.encode_observation(obs)
        action_mask = env.action_space.build_action_mask(obs)
        print("obs_vector shape:", obs_vector.shape)
        print("action_mask shape:", action_mask.shape)
        print("action_dim:", env.action_space.get_action_dim())

        action_id = env.action_space.encode(args.debug_action)
        next_obs, reward, done, step_info = env.step(action_id)
        print("step succeeded")
        print("debug_action:", args.debug_action)
        print("reward:", reward)
        print("done:", done)
        print("next_candidate_count:", len(next_obs.get("candidate_elements", [])))
        print("last_action_error:", step_info.get("last_action_error"))
        if args.debug_action == "inspect_layout":
            _print_layout_debug(args.site_id, obs, next_obs, {"action_type": args.debug_action, "action_id": action_id}, step_info)
        return 0
    finally:
        env.close()


def _print_candidate_debug(obs: dict) -> None:
    debug = obs.get("candidate_debug", {})
    print("[jaws-env-test] raw_axtree_nodes:", debug.get("raw_axtree_nodes", 0))
    print("[jaws-env-test] raw_extra_props:", debug.get("raw_extra_props", 0))
    print("[jaws-env-test] raw_dom_nodes:", debug.get("raw_dom_nodes", 0))
    print("[jaws-env-test] candidate_source_counts:", debug.get("candidate_source_counts", {}))
    print("[jaws-env-test] catalog_candidate_count:", debug.get("catalog_candidate_count", 0))
    print("[jaws-env-test] catalog_keyword_match_count:", debug.get("catalog_keyword_match_count", 0))
    print("[jaws-env-test] catalog_selector_match_count:", debug.get("catalog_selector_match_count", 0))
    if debug.get("catalog_candidate_count", 0) and not debug.get("catalog_selector_match_count", 0):
        print("WARNING: bug catalog is loaded but no selector matched; check selectors or rely on keyword-guided candidates.")
    print("[jaws-env-test] openended_interactive_candidate_count:", debug.get("openended_interactive_candidate_count", 0))
    print("[jaws-env-test] openended_keyword_match_count:", debug.get("openended_keyword_match_count", 0))
    print("[jaws-env-test] rejected_counts:", debug.get("rejected_counts", {}))


def _print_top_candidates(obs: dict) -> None:
    candidates = obs.get("candidate_elements", []) or []
    if not candidates:
        print("[jaws-env-test] no candidates extracted; see candidate_debug above")
        return
    print("[jaws-env-test] top candidates:")
    for index, candidate in enumerate(candidates[:20]):
        text = candidate.get("text") or candidate.get("name") or ""
        print(
            "  "
            f"{index}: "
            f"bid={candidate.get('bid')} "
            f"role={candidate.get('role')} "
            f"tag={candidate.get('tag')} "
            f"text={_safe(str(text)[:80])} "
            f"data_bug_id={candidate.get('data_bug_id')} "
            f"selector_hint={candidate.get('selector_hint')} "
            f"source={candidate.get('source')} "
            f"visibility={candidate.get('visibility')} "
            f"clickable={candidate.get('clickable')} "
            f"bbox={candidate.get('bbox')} "
            f"clickable_score={candidate.get('clickable_score')} "
            f"action_priority={candidate.get('action_priority')} "
            f"catalog_keyword_matches={candidate.get('catalog_keyword_matches')} "
            f"catalog_bug_id_matches={candidate.get('catalog_bug_id_matches')} "
            f"catalog_selector_match={candidate.get('catalog_selector_match')} "
            f"action_hints={candidate.get('action_hints')} "
            f"catalog_action_priority={candidate.get('catalog_action_priority')} "
            f"is_workout_add_action={candidate.get('is_workout_add_action')} "
            f"is_weekly_stats_related={candidate.get('is_weekly_stats_related')} "
            f"is_empty_state_related={candidate.get('is_empty_state_related')} "
            f"is_chart_related={candidate.get('is_chart_related')} "
            f"is_layout_target={candidate.get('is_layout_target')} "
            f"layout_check_type={candidate.get('layout_check_type')} "
            f"is_interactive={candidate.get('is_interactive')} "
            f"is_form_field={candidate.get('is_form_field')} "
            f"is_login_related={candidate.get('is_login_related')} "
            f"is_cart_related={candidate.get('is_cart_related')} "
            f"is_checkout_related={candidate.get('is_checkout_related')} "
            f"is_search_related={candidate.get('is_search_related')} "
            f"is_sparse_related={candidate.get('is_sparse_related')} "
            f"is_forbidden_related={candidate.get('is_forbidden_related')} "
            f"is_async_related={candidate.get('is_async_related')} "
            f"is_hang_related={candidate.get('is_hang_related')} "
            f"is_cart_quantity_related={candidate.get('is_cart_quantity_related')} "
            f"is_quantity_control={candidate.get('is_quantity_control')} "
            f"openended_keyword_matches={candidate.get('openended_keyword_matches')} "
            f"openended_action_priority={candidate.get('openended_action_priority')}"
        )


def _print_layout_debug(site_id: str, before_obs: dict, after_obs: dict, action: dict, step_info: dict) -> None:
    known_bugs = load_known_bugs(site_id)
    site_profile = build_site_profile(site_id, known_bugs, exploration_profile="openended_commerce" if not known_bugs else None)
    anomalies = detect_anomalies(
        before_obs,
        after_obs,
        {"action": action, "site_profile": site_profile, **step_info},
        site_profile=site_profile,
    )
    matches = match_anomalies_to_known_bugs(anomalies, known_bugs, site_id=site_id)
    layout_signals = after_obs.get("layout_signals", {}) or {}
    layout_anomalies = [item for item in anomalies if item.get("type") == "layout-overflow"]
    print("[jaws-env-test] layout_signals:", layout_signals)
    print("[jaws-env-test] layout_overflow_candidates_raw:", len(layout_anomalies))
    print("[jaws-env-test] layout_overflow_matches:", [match.get("matched_bug_id") for match in matches if match.get("type") == "layout-overflow"])
    for index, anomaly in enumerate(layout_anomalies[:5]):
        print(f"[jaws-env-test] layout_anomaly_{index}_evidence:", anomaly.get("evidence", {}))
        print(f"[jaws-env-test] layout_anomaly_{index}_match:", anomaly.get("matched_bug_id"), anomaly.get("match_score"), anomaly.get("match_reason"))


def _warn_if_url_mismatch(requested_url: str, observed_url: object) -> None:
    requested = urlparse(requested_url)
    observed = urlparse(str(observed_url or ""))
    if requested.netloc and observed.netloc and requested.netloc != observed.netloc:
        print(f"WARNING: requested base_url is {requested_url} but BrowserGym opened {observed_url}")


def _safe(value: object) -> str:
    return str(value).encode("unicode_escape", errors="backslashreplace").decode("ascii")


if __name__ == "__main__":
    raise SystemExit(main())
