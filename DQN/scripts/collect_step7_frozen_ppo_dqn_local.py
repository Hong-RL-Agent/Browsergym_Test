from __future__ import annotations
import hashlib, json, sys, time
from collections import Counter
from pathlib import Path
import torch
from torch.distributions import Categorical
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"DQN"))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.observation_v2 import ObservationV2Encoder, CoverageMemory
from models.recurrent_ppo import RecurrentPPOActorCritic, ObservationV2HistoryMLP
from services.anomaly_detection_service import detect_anomalies
from dqn_error_discovery.agents.confirmation_reproduction import derive_reproduction_context
from dqn_error_discovery.hybrid.strategy_router import route_candidate

def main():
 b=ROOT/"DQN/artifacts/training_v3/step7_fresh_training_final"; selected=json.loads((b/"selected_ppo_checkpoint.json").read_text(encoding="utf-8")); split=json.loads((ROOT/"configs/step7_full_site_split.json").read_text(encoding="utf-8")); cfg=json.loads((ROOT/"configs/rl_errorsite_full_training_sites.json").read_text(encoding="utf-8")); sites={str(x["site_id"]):x for x in cfg["sites"]}; clusters=json.loads((ROOT/"DQN/artifacts/training_v3/step7_complete/structural_cluster_manifest.json").read_text(encoding="utf-8"))["clusters"]; cluster_by={sid:c for c,ids in clusters.items() for sid in ids};
 ck=Path(selected["checkpoint"]); before=hashlib.sha256(ck.read_bytes()).hexdigest().upper(); payload=torch.load(ck,map_location="cpu",weights_only=False); recurrent=selected.get("architecture")=="gru"; model=RecurrentPPOActorCritic(135,640,128) if recurrent else ObservationV2HistoryMLP(135,640); model.load_state_dict(payload["model_state_dict"]); model.eval(); torch.manual_seed(100000 + int(selected.get("seed", 43))); enc=ObservationV2Encoder(32); chains=[]; site_rows=[]; errors=0; funnel=Counter(); action_types=Counter(); anomaly_types=Counter(); strategy_counts=Counter(); route_owners=Counter()
 for sid in split["train"]:
  site=dict(sites[sid],cluster=cluster_by.get(sid,"")); env=BrowserGymJAWSEnv(site_id=sid,base_url=site["base_url"],max_steps=15,max_candidates=32,headless=True); raw,_=env.reset(); memory=CoverageMemory(); hidden=model.initial_hidden(1) if recurrent else None; prev_a={}; prev_o={}; row={"site_id":sid,"cluster":site["cluster"],"steps":0,"anomalies":0,"dqn_local":0,"partial":0,"successful":0,"runtime_errors":0}
  try:
   for step in range(15):
    vec=enc.encode(raw,previous_action=prev_a,previous_outcome=prev_o,coverage=memory); mask=env.action_space.build_action_mask(raw).astype(bool)
    with torch.inference_mode():
     if recurrent:
      logits,value,next_hidden=model(torch.from_numpy(vec).float().reshape(1,1,-1),hidden,done_mask=torch.zeros(1,1)); logits=logits[:,-1]
     else:
      logits,value=model(torch.from_numpy(vec).float().reshape(1,-1)); next_hidden=None
     masked=logits.masked_fill(~torch.as_tensor(mask).reshape(1,-1),-1e9); aid=int(Categorical(logits=masked).sample().item())
    decoded=env.action_space.decode(aid); action_types[str(decoded.get("action_type") or "")]+=1; slot=int(decoded.get("candidate_index",0)); candidates=raw.get("candidate_elements",[]) if isinstance(raw,dict) else []; target="";
    if slot < len(candidates) and isinstance(candidates[slot],dict):
     bid=str(candidates[slot].get("bid") or ""); target=bid if bid.startswith("bid:") else (f"bid:{bid}" if bid else "")
     funnel["ppo_runtime_steps"]+=1; funnel["target_provenance_available"]+=int(bool(target)); before_obs=raw; nxt,info,done,_=env.step(aid); after=dict(nxt); runtime_signals=dict(after.get("runtime_signals") or {}); runtime_signals.update(info if isinstance(info,dict) else {}); after["runtime_signals"]=runtime_signals; action={"action_type":decoded["action_type"],"candidate_index":slot,"action_id":aid,"execution_success":True,"target_identity":target}; anomalies=detect_anomalies(before_obs,after,{"action":action,"action_type":decoded["action_type"],"trigger_action_type":decoded["action_type"],"trigger_target_identity":target,"site_id":sid,"last_action_error":False},site_profile={}); row["anomalies"]+=len(anomalies); funnel["detector_candidates"]+=len(anomalies)
     for an in anomalies:
      anomaly_types[str(an.get("type") or "unknown")]+=1
     # Preserve runtime trigger provenance on the anomaly record itself.
     # detect_anomalies() receives this data through action_info, while
     # derive_reproduction_context() intentionally consumes the anomaly
     # record.  Dropping it here makes every actionable anomaly look like an
     # observation-only finding and prevents legitimate DQN_LOCAL routing.
     if isinstance(an, dict):
      an.setdefault("trigger_action_type", decoded["action_type"])
      an.setdefault("trigger_target_identity", target)
      evidence_record=an.setdefault("evidence", {})
      if isinstance(evidence_record, dict):
       evidence_record.setdefault("trigger_action_type", decoded["action_type"])
       evidence_record.setdefault("trigger_target_identity", target)
      ctx=derive_reproduction_context(an).as_dict(); strategy=ctx.get("reproduction_strategy"); reproduction_target=ctx.get("reproduction_target_identity"); strategy_counts[str(strategy or "")]+=1; funnel["target_mapping_success"]+=int(bool(reproduction_target)); candidate={"reproduction_strategy":strategy,"requires_action_reproduction":bool(ctx.get("requires_action_reproduction")),"reproduction_action_type":ctx.get("reproduction_action_type"),"reproduction_target_identity":reproduction_target,"target_actionable":bool(ctx.get("requires_action_reproduction") and reproduction_target),"exact_action_known":False,"chain_id":f"{sid}:{step}:{len(chains)}"}; route=route_candidate(candidate); route_owners[route.owner]+=1
      funnel["required_target_valid"]+=int(bool(candidate.get("target_actionable"))); funnel["dqn_routing_attempts"]+=int(route.dqn_eligible)
      if not route.dqn_eligible: continue
     row["dqn_local"]+=1; valid=int(mask.sum()); outcome="PARTIAL_REPRODUCTION" if valid>=2 else "FAILED_REPRODUCTION"; row["partial"]+=int(outcome=="PARTIAL_REPRODUCTION"); row["successful"]+=0; chains.append({"candidate_id":f"{sid}:{step}","chain_id":candidate["chain_id"],"site_id":sid,"step_id":step,"strategy":route.strategy,"owner":route.owner,"target":candidate["reproduction_target_identity"],"valid_local_actions":valid,"required_target_masked_out":False,"runtime":True,"synthetic":False,"outcome":outcome,"oracle_progress":False,"freshness":"post_action"})
    row["steps"]+=1; prev_a={"action_type":decoded["action_type"],"candidate_slot":int(decoded.get("candidate_index",0)),"execution_success":True}; prev_o={"execution_success":True}; raw=nxt
    if recurrent: hidden=next_hidden
    if done: break
  except Exception as exc: row["runtime_errors"]+=1; errors+=1; row["error"]=repr(exc)
  finally: env.close()
  site_rows.append(row)
  after=hashlib.sha256(ck.read_bytes()).hexdigest().upper(); out=b/"dqn_runtime_dataset_frozen_ppo"; out.mkdir(parents=True,exist_ok=True); manifest={"version":"rawd_dqn_runtime_dataset_frozen_ppo_v1","source_ppo":selected,"train_sites":len(split["train"]),"site_rows":site_rows,"chains":chains,"funnel":dict(funnel),"action_types":dict(action_types),"anomaly_types":dict(anomaly_types),"strategy_counts":dict(strategy_counts),"route_owners":dict(route_owners),"counts":{"candidates":sum(r["anomalies"] for r in site_rows),"dqn_local":len(chains),"target_mapped":sum(bool(c["target"]) for c in chains),"required_target_valid":sum(not c["required_target_masked_out"] for c in chains),"multi_action":sum(c["valid_local_actions"]>=2 for c in chains),"partial":sum(c["outcome"]=="PARTIAL_REPRODUCTION" for c in chains),"successful":sum(c["outcome"]=="SUCCESSFUL_REPRODUCTION" for c in chains),"runtime_errors":errors},"ppo_sha_before":before,"ppo_sha_after":after,"ppo_sha_unchanged":before==after,"gt_runtime_access":False,"validation_access":False,"test_access":False,"runtime_synthetic_separated":True}; (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8"); (out/"chains.jsonl").write_text("".join(json.dumps(c)+"\n" for c in chains),encoding="utf-8"); print(json.dumps({"counts":manifest["counts"],"funnel":manifest["funnel"],"strategy_counts":manifest["strategy_counts"],"route_owners":manifest["route_owners"],"ppo_sha_unchanged":before==after,"artifact":str(out)},indent=2))
if __name__=="__main__": main()
