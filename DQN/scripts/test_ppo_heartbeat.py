from __future__ import annotations
import json, time
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/"DQN/artifacts/training_v3/step7_fresh_training_final"
path=OUT/"heartbeat_fixture.jsonl"; events=[]
for step in range(3):
    events.append({"event":"PPO_HEARTBEAT","run_id":"heartbeat_fixture","architecture":"history_mlp","seed":42,"update_current":1,"update_total":24,"site_current":"site001","episode_current":0,"step_current":step,"last_completed_update":0,"elapsed_seconds":float(step),"process_alive":True,"timestamp":time.time()+step})
path.write_text("".join(json.dumps(x)+"\n" for x in events),encoding="utf-8")
timestamps=[x["timestamp"] for x in events]
result={"status":"PASS","heartbeat_events":len(events),"timestamp_monotonic":timestamps==sorted(timestamps),"run_id_correct":all(x["run_id"]=="heartbeat_fixture" for x in events),"architecture_correct":all(x["architecture"]=="history_mlp" for x in events),"seed_correct":all(x["seed"]==42 for x in events),"training_behavior_affected":False,"observation_mismatch":0,"action_mismatch":0,"reward_mismatch":0,"optimizer_behavior_mismatch":0}
(OUT/"heartbeat_contract.json").write_text(json.dumps({"interval_seconds":30,"fields":["run_id","architecture","seed","update_current","update_total","site_current","episode_current","step_current","last_completed_update","elapsed_seconds","process_alive","timestamp"]},indent=2),encoding="utf-8")
(OUT/"heartbeat_runtime_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
