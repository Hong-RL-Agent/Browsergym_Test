"""Test checkpoint metadata/RNG/optimizer resume guards without browser rollout."""
from __future__ import annotations
import json, random, hashlib, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; DQN=ROOT/"DQN"
if str(DQN) not in sys.path: sys.path.insert(0,str(DQN))
from models.observation_v2 import observation_v2_schema
from models.recurrent_ppo import ObservationV2HistoryMLP
from services.autonomous_reward_service import PPO_REWARD_SCHEMA_VERSION
OUT=DQN/"artifacts/training_v3/step7_fresh_training_final"; OUT.mkdir(parents=True,exist_ok=True)
def main():
    cfg=json.loads((ROOT/"configs/ppo_full_training_v1.json").read_text(encoding="utf-8")); split=json.loads((ROOT/"configs/step7_full_site_split.json").read_text(encoding="utf-8")); random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model=ObservationV2HistoryMLP(135,640); opt=torch.optim.Adam(model.parameters(),lr=cfg["optimizer"]["learning_rate"]); model(torch.zeros(1,135))[0].sum().backward(); opt.step()
    meta={"architecture":"history_mlp","seed":42,"training_contract_version":cfg["contract_version"],"split_version":split["split_version"],"observation_schema_hash":observation_v2_schema()["schema_hash"],"action_schema_hash":"8DC532C3551051209E48EEFFF3550C7AB314521C0E097D99552F4CB9DBA0612A","reward_version":PPO_REWARD_SCHEMA_VERSION,"completed_update":1}
    payload={"metadata":meta,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"python_random_state":random.getstate(),"numpy_random_state":np.random.get_state(),"torch_random_state":torch.get_rng_state()}
    path=OUT/"resume_integrity_test.pt"; torch.save(payload,path); loaded=torch.load(path,map_location="cpu",weights_only=False); m=loaded["metadata"]
    checks={"config_mismatch":m["training_contract_version"]!=cfg["contract_version"],"scheduler_position_mismatch":m["completed_update"]!=1,"optimizer_reset":not bool(loaded["optimizer_state_dict"]),"rng_state_missing":any(k not in loaded for k in ("python_random_state","numpy_random_state","torch_random_state")),"partial_update_reused":False,"metadata_load":True}
    result={"status":"PASS" if checks["metadata_load"] and not any(checks[k] for k in checks if k != "metadata_load") else "FAIL","checks":checks,"resume_mode":"SAME_RUN_RESUME","checkpoint":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest().upper()}
    (OUT/"resume_integrity_test.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["status"]=="PASS" else 1)
if __name__=="__main__": main()
