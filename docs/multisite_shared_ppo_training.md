# Multi-Site Shared PPO Training

## Why Shared PPO

The previous BrowserGym PPO workflow saved one checkpoint per site, for example:

- `artifacts/models/site001_browsergym_ppo.pt`
- `artifacts/models/site003_browsergym_ppo.pt`

That is useful for isolated experiments, but it weakens generalization. A policy trained only on one website can overfit to that site's labels, layout, and action order. When a new website is added, the agent does not automatically reuse interaction patterns learned elsewhere, such as no-response clicks, empty-state contradictions, duplicated rendering, or layout overflow.

The multi-site workflow keeps one shared Actor-Critic policy and trains it with rollouts from multiple BrowserGym openended environments.

## Shared Model Structure

The shared checkpoint is configured in `configs/training_sites.json`:

```json
{
  "shared_model_path": "artifacts/models/jaws_browsergym_shared_ppo.pt",
  "output_dir": "artifacts/multisite",
  "sites": []
}
```

All sites use the same `PPOAgent`, `ObservationEncoder`, and `ActionSpace`. Site environments are separate, but policy weights are shared.

## Rollout Flow

Training runs in update rounds:

1. For each update, iterate through configured sites.
2. Collect `episodes_per_site` BrowserGym episodes from each site.
3. Compute returns and advantages per episode.
4. Merge all episode buffers into one shared PPO rollout buffer.
5. Run one PPO update on the shared agent.
6. Save the shared checkpoint.
7. Write site-specific logs and summaries.

This keeps site experience separated for reporting while accumulating policy learning into one model.

## Known Catalog vs Openended Sites

Sites with a known bug catalog, such as `site001` and `site003`, report:

- `matched_bug_ids`
- `missed_bug_ids`
- `precision`
- `recall`

Sites without a catalog can be added later and report openended anomaly discovery metrics:

- `unique_detected_candidates`
- `total_detected_candidates`
- anomaly type counts
- action distribution

Precision and recall are `null` for sites without known ground truth. `site9800` was first explored in open-ended mode and is now configured to use `datasets/site9800/bug_catalog.json` when that catalog is present.

Openended sites can specify an `exploration_profile` in `configs/training_sites.json`. `site9800` uses `openended_commerce`, which prioritizes generic commerce interactions such as login, search, cart, checkout, filters, forms, submit buttons, and product action buttons. This profile remains only an exploration hint; known-bug precision/recall come from `datasets/site9800/bug_catalog.json`.

For `site9800`, open-ended findings should still be reviewed before promotion into the confirmed catalog. The review checklist is maintained in `docs/site9800_openended_review.md`, and unconfirmed candidates are stored in `datasets/site9800/bug_catalog.candidates.json`. Add/cart navigation false positives should remain excluded when URL, page text, or cart state changes prove the click had an effect.

The multi-site evaluator automatically refreshes both review artifacts for `site9800`. If you already saved an evaluation JSON file and only want to regenerate the review artifacts, run:

```powershell
.\BrowserGym\.venv\Scripts\python.exe scripts\export_site9800_review.py --evaluation-json artifacts\multisite\latest_eval.json
```

## Site Identity Preflight

The `site_id` and `base_url` mapping must be correct before training. Current local port mapping:

- `site001`: `http://localhost:9220`
- `site002`: `http://localhost:9221`
- `site003`: `http://localhost:9222`
- `site9800`: `http://localhost:9800` (known catalog mode when `datasets/site9800/bug_catalog.json` exists)

The default multi-site config trains `site001`, `site003`, and `site9800`. `site002` should only be added as a separate `site_id` when `datasets/site002/bug_catalog.json` exists.

Smoke tests print `data_bug_ids_found`, `expected_bug_id_prefix`, and `site_identity_match`. If the requested `site_id` is `site003`, observed data bug IDs must start with `site003-bug`. If they start with `site002-bug`, the model is not failing; the config is opening the wrong website.

Training runs a preflight reset for every configured site. Use `--strict-site-validation true` to fail fast on connection failures or data-bug-id prefix mismatches. For sites with `has_bug_catalog: false`, preflight does not fail on missing `data-bug-id` values; `candidate_count > 0` is enough to proceed in openended mode.

## Reward Design

Generic rewards apply across all sites:

- click no-response signals
- DOM/text no-change after meaningful interaction
- layout overlap and overflow signals
- duplicated rendering
- empty-state contradiction
- action error and invalid action penalties
- repeated action penalties
- exploration reward

Site-specific rewards are limited and supplemental. Most target selection, anomaly matching, and reward shaping comes from each site's bug catalog so a new site can be added by writing catalog metadata rather than hardcoding strings.

## Commands

## site9800 Catalog Mode

`site9800` can run in two modes:

- Known bug catalog mode: `datasets/site9800/bug_catalog.json` exists and `configs/training_sites.json` has `has_bug_catalog: true`.
- Open-ended mode: no confirmed catalog is available, so precision/recall are `null` and exploratory anomalies are exported for human review.

The current `site9800` catalog is prepared from the ERRORS.md categories supplied for SmartCommerce Global: forbidden/API mismatch, sparse data rendering, ASYNC/HANG, timeout/no feedback, and cart quantity mismatch. If an authoritative `ERRORS.md` is added later, update `datasets/site9800/bug_catalog.json` with its exact selectors, trigger text, and expected evidence.

The previous open-ended candidates C1-C4 are not treated as confirmed bugs:

- Dashboard/search textbox click alone is not a form-no-feedback bug.
- Checkout/search with validation, error, empty-state, or other feedback is not a no-feedback bug.
- Repeated Add buttons across product cards are normal repeated UI unless the same component overlaps or duplicates inside the same parent/container.
- Same-page Cart clicks are marked likely false positive unless a cart quantity/count mismatch is observed.

Pre-check each site:

```powershell
.\BrowserGym\.venv\Scripts\python.exe scripts\test_browsergym_jaws_env.py --site-id site001 --base-url http://localhost:9220
.\BrowserGym\.venv\Scripts\python.exe scripts\test_browsergym_jaws_env.py --site-id site003 --base-url http://localhost:9222
.\BrowserGym\.venv\Scripts\python.exe scripts\test_browsergym_jaws_env.py --site-id site9800 --base-url http://localhost:9800
```

Train the shared model:

```powershell
.\BrowserGym\.venv\Scripts\python.exe runners\train_multisite_browsergym_agent.py --config configs\training_sites.json --total-updates 20 --episodes-per-site 1 --max-steps 25
```

Strict preflight mode:

```powershell
.\BrowserGym\.venv\Scripts\python.exe runners\train_multisite_browsergym_agent.py --config configs\training_sites.json --strict-site-validation true
```

Resume from a checkpoint:

```powershell
.\BrowserGym\.venv\Scripts\python.exe runners\train_multisite_browsergym_agent.py --config configs\training_sites.json --load-model artifacts\models\jaws_browsergym_shared_ppo.pt
```

Resume and save back to the shared checkpoint:

```powershell
.\BrowserGym\.venv\Scripts\python.exe runners\train_multisite_browsergym_agent.py --config configs\training_sites.json --total-updates 20 --episodes-per-site 1 --max-steps 25 --load-model artifacts\models\jaws_browsergym_shared_ppo.pt --save-model artifacts\models\jaws_browsergym_shared_ppo.pt
```

Evaluate the shared model:

```powershell
.\BrowserGym\.venv\Scripts\python.exe runners\evaluate_multisite_browsergym_agent.py --config configs\training_sites.json --model-path artifacts\models\jaws_browsergym_shared_ppo.pt
```

Single-site runners remain available for isolated experiments:

```powershell
.\BrowserGym\.venv\Scripts\python.exe runners\train_browsergym_agent.py --site-id site001 --base-url http://localhost:9220
.\BrowserGym\.venv\Scripts\python.exe runners\evaluate_browsergym_agent.py --site-id site001 --base-url http://localhost:9220 --model-path artifacts\models\site001_browsergym_ppo.pt
```

## Output Files

Shared outputs:

- `artifacts/models/jaws_browsergym_shared_ppo.pt`
- `artifacts/multisite/multisite_training_summary.json`

Per-site outputs:

- `artifacts/multisite/{site_id}/training_summary.json`
- `artifacts/multisite/{site_id}/detected_bugs.json`
- `artifacts/multisite/{site_id}/rl_transition_log.jsonl`

## Future Parallel Workers

The current implementation collects site episodes sequentially to keep BrowserGym and Playwright resource usage predictable. A later extension can run separate worker processes per site, then merge serialized rollout buffers before PPO update. The key invariant should remain unchanged: workers may have separate environments, but the PPO update writes to one shared policy checkpoint.
