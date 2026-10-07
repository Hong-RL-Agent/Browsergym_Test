# DEV_NEXT Async Testbed

Development-only framework behavior testbed for frozen Detector v1 false-positive characterization.

- Framework: Next.js behavior fixture
- Version metadata: actual Next package is not installed in the local repository; this fixture reproduces controlled Next-style route transition, `_next` static resource, prefetch, and RSC-like request behavior without adding external dependencies.
- Startup: `npm start`
- URL: `http://localhost:9226/`
- Health: `http://localhost:9226/api/health`
- Scenario catalog: `SCENARIOS.json`
- Ground truth policy: post-hoc evaluation only. Do not pass scenario metadata into PPO, DQN, Detector, reward, candidate generation, or Hybrid Controller runtime.

This site is not part of SITE001~SITE005 and must not be used as a final evaluation site.
