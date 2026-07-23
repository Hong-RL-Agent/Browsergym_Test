# PPO vs. Masked Rainbow DQN and OWASP-safe API tests

## Equal-budget comparison

Both policies are trained and evaluated with the same site, seed, episode count,
and maximum steps. The default selection order is recall, precision, average
reward, and then fewer false positives. A tie keeps PPO as the conservative
default. One site and one seed are a smoke comparison, not a final benchmark.

```powershell
.\.venv\Scripts\python.exe scripts\run_algorithm_ab.py `
  --site-id site001 `
  --base-url http://127.0.0.1:9220 `
  --train-episodes 10 `
  --eval-episodes 3 `
  --max-steps 20 `
  --seed 42
```

Masked Rainbow DQN includes action masking, Double DQN target selection,
dueling value/advantage streams, C51 distributional values, proportional
prioritized replay, three-step returns, and noisy linear exploration.

## OWASP-safe API tests

The default catalog is `configs/owasp_safe_test_catalog.json`. Only listed
mutations run. High-load, repeated, parallel, and long-string mutations are
disabled by default. Non-GET requests require all three controls:

1. `--allow-mutating` on the runner.
2. `allow_mutating_requests: true` on the test site.
3. `test_safe: true` on the endpoint.

This is intended only for systems the operator is authorized to test.

```powershell
.\.venv\Scripts\python.exe runners\run_api_fuzz_eval.py `
  --config configs\api_sites.json
```
