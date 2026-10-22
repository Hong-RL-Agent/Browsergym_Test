from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"DQN/artifacts/training_v3/step7_fresh_training_final"; sys.path.insert(0,str(ROOT/"DQN"))
from models.recurrent_ppo import RecurrentPPOActorCritic, ObservationV2HistoryMLP
from models.observation_v2 import observation_v2_schema
from models.action_types import action_space_mapping_hash
RUNS=[("history_mlp",42,"fresh5_mlp42_rewardfix"),("history_mlp",43,"fresh5_mlp43_rewardfix"),("history_mlp",44,"fresh5_mlp44_rewardfix"),("gru",42,"fresh5_gru42_rewardfix"),("gru",43,"fresh5_gru43_rewardfix"),("gru",44,"fresh5_gru44_rewardfix")]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def main():
 v=json.loads((OUT/"ppo_validation_results_canonical.json").read_text(encoding="utf-8")); rows=v["per_run"]
 def key(r): return (-r["functional_state_coverage"]["mean"],-r["flow_coverage"]["mean"],-r["exploration_diversity"]["mean"],-r["candidate_discovery"]["mean"],-r["target_coverage"]["mean"],r["repeat_rate"]["mean"],r["no_effect_rate"]["mean"],r["runtime_error_rate"]["mean"],r["seed"])
 eligible=[r for r in rows if r.get("integrity_gate")=="PASS" and r.get("functional_exploration_gate",{}).get("status")=="PASS"]
 if not eligible: raise RuntimeError("No validation-eligible fresh PPO checkpoint")
 selected=sorted(eligible,key=key)[0]; arch,seed,prefix=next(x for x in RUNS if x[0]==selected["architecture"] and x[1]==selected["seed"]); ck=OUT/f"{prefix}_full.pt"; meta=json.loads((OUT/f"{prefix}_metadata.json").read_text(encoding="utf-8")); payload=torch.load(ck,map_location="cpu",weights_only=False); model=RecurrentPPOActorCritic(135,640,128) if arch=="gru" else ObservationV2HistoryMLP(135,640)
 model.load_state_dict(payload["model_state_dict"]); model.eval()
 if arch=="gru": h=model.initial_hidden(1); logits,val,nh=model(torch.zeros(1,1,135),h,done_mask=torch.zeros(1,1))
 else: logits,val=model(torch.zeros(1,135))
 reload_pass=bool(logits.shape[-1]==640 and torch.isfinite(logits).all() and torch.isfinite(val).all())
 result={"schema_version":"rawd_step7_frozen_ppo_v1","architecture":arch,"seed":seed,"run_id":meta.get("run_id",prefix),"checkpoint":str(ck),"checkpoint_file_sha256":sha(ck),"checkpoint_metadata_sha256":meta.get("checkpoint_metadata_sha256"),"observation_dimension":135,"observation_hash":observation_v2_schema()["schema_hash"],"action_dimension":640,"action_hash":action_space_mapping_hash(),"reward_version":meta.get("reward_version"),"training_contract":meta.get("training_contract_version"),"split_version":meta.get("split_version"),"validation_metrics":selected,"selection_rule":["functional_state_coverage","flow_coverage","exploration_diversity","candidate_discovery","target_coverage","lower_repeat_rate","lower_no_effect_rate","runtime_stability"],"selection_rationale":"lexicographic frozen rule over canonical validation metrics; TEST not accessed","validation_optimizer_steps":0,"validation_weight_mutations":0,"gt_runtime_access":False,"known_bug_runtime_access":False,"frozen":True,"clean_reload":reload_pass}
 (OUT/"selected_ppo_checkpoint.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8"); print(json.dumps({"selected":{k:result[k] for k in ("architecture","seed","run_id","checkpoint","checkpoint_file_sha256")},"clean_reload":reload_pass,"frozen":True},indent=2))
if __name__=="__main__": main()
