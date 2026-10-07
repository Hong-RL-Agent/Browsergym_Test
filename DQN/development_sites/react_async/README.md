# DEV_REACT Async Testbed

Development-only React async lifecycle testbed for frozen Detector v1 false-positive characterization.

- Framework: React 18.2.0, production-like UMD bundle
- StrictMode: disabled
- Startup: `npm start`
- URL: `http://localhost:9225/`
- Health: `http://localhost:9225/api/health`
- Scenario catalog: `SCENARIOS.json`
- Ground truth policy: post-hoc evaluation only. Do not pass scenario metadata into PPO, DQN, Detector, reward, candidate generation, or Hybrid Controller runtime.

This site is not part of SITE001~SITE005 and must not be used as a final evaluation site.
