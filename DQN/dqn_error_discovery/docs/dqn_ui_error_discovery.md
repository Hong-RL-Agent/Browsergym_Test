# UI/runtime error discovery with masked DQN

The DQN runner is an opt-in QA workflow for ordinary BrowserGym UI and runtime
errors. It reuses the repository observation encoder and anomaly detector.

The runner excludes infrastructure, authorization, and security inspection
actions. It is intended for findings such as no-response interactions, layout
issues, and duplicated rendering. It does not perform security testing or
vulnerability reproduction.

```powershell
python -m dqn_error_discovery.runners.train_dqn_error_detector `
  --base-url http://127.0.0.1:9220 `
  --site-id site001 `
  --episodes 20 `
  --output artifacts/models/browsergym_ui_dqn.pt
```

The checkpoint manifest stores aggregate finding counts and a target URL hash;
it does not store credentials, request payloads, or security findings.
