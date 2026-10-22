from __future__ import annotations
import hashlib, json, sys, statistics, os, time
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.observation_v2 import ObservationV2Encoder, CoverageMemory, functional_state_fingerprint, observation_v2_schema
from models.recurrent_ppo import RecurrentPPOActorCritic, ObservationV2HistoryMLP
from models.action_types import action_space_mapping_hash
from services.autonomous_reward_service import calculate_autonomous_reward

OUT = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
RUNS = [
    ("history_mlp", 42, "fresh5_mlp42_rewardfix"), ("history_mlp", 43, "fresh5_mlp43_rewardfix"), ("history_mlp", 44, "fresh5_mlp44_rewardfix"),
    ("gru", 42, "fresh5_gru42_rewardfix"), ("gru", 43, "fresh5_gru43_rewardfix"), ("gru", 44, "fresh5_gru44_rewardfix"),
]

def digest(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def mean(xs): return float(np.mean(xs)) if xs else 0.0
def median(xs): return float(np.median(xs)) if xs else 0.0
def std(xs): return float(np.std(xs)) if xs else 0.0

def evaluate(model, recurrent, site, steps=15, selector="sample"):
    env = BrowserGymJAWSEnv(site_id=site["site_id"], base_url=site["base_url"], max_steps=steps, max_candidates=32, headless=True)
    enc = ObservationV2Encoder(32); memory = CoverageMemory(); prev_a={}; prev_o={}; hidden=model.initial_hidden(1) if recurrent else None
    raw,_=env.reset(); rows=[]; states=set(); targets=set(); actions=set(); sa=set(); sao=set(); repeats=0; no_effect=0; invalid=0; timeout=0; runtime_errors=0; functional_transitions=0; interaction_opportunity_steps=0; interaction_mask_valid_steps=0; click_opportunity_steps=0; click_mask_valid_steps=0; click_selected=0; fill_opportunity_steps=0; fill_mask_valid_steps=0; fill_selected=0; select_opportunity_steps=0; select_mask_valid_steps=0; select_selected=0; enter_opportunity_steps=0; enter_mask_valid_steps=0; enter_selected=0; viewport_only_steps=0; before_fp,_=functional_state_fingerprint(raw); states.add(before_fp)
    try:
        for step in range(steps):
            vec=enc.encode(raw, previous_action=prev_a, previous_outcome=prev_o, coverage=memory); mask=env.action_space.build_action_mask(raw).astype(bool); obs=torch.from_numpy(vec).float()
            click_valid=bool(mask[32:64].any()); fill_valid=bool(mask[448:480].any()); select_valid=bool(mask[480:512].any()); enter_valid=bool(mask[512:544].any()); interaction_valid=click_valid or fill_valid or select_valid or enter_valid
            interaction_opportunity_steps += int(interaction_valid); interaction_mask_valid_steps += int(interaction_valid); click_opportunity_steps += int(click_valid); click_mask_valid_steps += int(click_valid); fill_opportunity_steps += int(fill_valid); fill_mask_valid_steps += int(fill_valid); select_opportunity_steps += int(select_valid); select_mask_valid_steps += int(select_valid); enter_opportunity_steps += int(enter_valid); enter_mask_valid_steps += int(enter_valid)
            with torch.inference_mode():
                if recurrent:
                    logits,value,next_hidden=model(obs.reshape(1,1,-1),hidden,done_mask=torch.zeros((1,1))); logits=logits[:,-1]; value=value[:,-1]
                else: logits,value=model(obs.reshape(1,-1)); next_hidden=None
            masked=logits.masked_fill(~torch.as_tensor(mask).reshape(1,-1),-1e9)
            aid=int(torch.argmax(masked,dim=-1).item()) if selector == "argmax" else int(Categorical(logits=masked).sample().item())
            invalid += int(not bool(mask[aid])); decoded=env.action_space.decode(aid); cand=raw.get("candidate_elements",[]) if isinstance(raw,dict) else []; slot=int(decoded.get("candidate_index",0)); target=""
            if slot < len(cand) and isinstance(cand[slot],dict):
                bid=str(cand[slot].get("bid") or ""); target=bid if bid.startswith("bid:") else (f"bid:{bid}" if bid else "")
            action={"action_type":decoded["action_type"],"candidate_index":slot,"action_id":aid,"execution_success":True,"target_identity":target}; before,_=functional_state_fingerprint(raw); nxt,info,done,_=env.step(aid); after=dict(nxt); runtime_signals=dict(after.get("runtime_signals") or {}); runtime_signals.update(info if isinstance(info,dict) else {}); after["runtime_signals"]=runtime_signals; reward,comps=calculate_autonomous_reward(raw,after,action,[],[],{},{}); after_fp,_=functional_state_fingerprint(after); changed=after_fp!=before; cov=memory.observe(after_fp,action,comps)
            states.add(after_fp); actions.add(decoded["action_type"]); targets.add(target) if target else None; key=(before,aid); outcome_key=(before,aid,after_fp); repeats += int(key in sa); no_effect += int(not changed); selected_type=decoded["action_type"]; click_selected += int(selected_type=="click_element"); fill_selected += int(selected_type=="fill_input"); select_selected += int(selected_type=="select_option"); enter_selected += int(selected_type=="press_enter"); viewport_only_steps += int(selected_type in {"change_viewport_mobile","change_viewport_desktop"}); functional_transitions += int(changed and selected_type not in {"change_viewport_mobile","change_viewport_desktop"}); sa.add(key); sao.add(outcome_key); rows.append({"site_id":site["site_id"],"cluster":site["cluster"],"step":step,"action_id":aid,"action_type":selected_type,"target":target,"reward":float(reward),"state_changed":changed,"mask_valid":bool(mask[aid]),"interaction_mask_valid":interaction_valid,"done":bool(done),"components":comps,"coverage":cov}); prev_a={"action_type":selected_type,"candidate_slot":slot,"execution_success":True}; prev_o={"execution_success":True}; raw=nxt
            if recurrent: hidden=next_hidden
            if done: break
        timeout=int(len(rows)>=steps and not rows[-1].get("done",False))
    except Exception as exc:
        runtime_errors=1; rows.append({"site_id":site["site_id"],"runtime_error":repr(exc)})
    finally: env.close()
    n=max(1,len(rows)); return {"site_id":site["site_id"],"cluster":site["cluster"],"episodes":1,"steps":len(rows),"functional_state_coverage":len(states),"flow_coverage":len(states),"target_coverage":len(targets),"action_type_coverage":len(actions),"state_action_coverage":len(sa),"state_action_outcome_coverage":len(sao),"candidate_discovery":len(targets),"exploration_diversity":len(actions)/n,"repeat_rate":repeats/n,"no_effect_rate":no_effect/n,"invalid_rate":invalid/n,"timeout_rate":timeout,"runtime_error_rate":runtime_errors,"runtime_errors":runtime_errors,"interaction_opportunity_steps":interaction_opportunity_steps,"interaction_mask_valid_steps":interaction_mask_valid_steps,"click_opportunity_steps":click_opportunity_steps,"click_mask_valid_steps":click_mask_valid_steps,"click_selected":click_selected,"fill_opportunity_steps":fill_opportunity_steps,"fill_mask_valid_steps":fill_mask_valid_steps,"fill_selected":fill_selected,"select_opportunity_steps":select_opportunity_steps,"select_mask_valid_steps":select_mask_valid_steps,"select_selected":select_selected,"enter_opportunity_steps":enter_opportunity_steps,"enter_mask_valid_steps":enter_mask_valid_steps,"enter_selected":enter_selected,"functional_state_transitions":functional_transitions,"viewport_only_steps":viewport_only_steps,"same_viewport_noop_count":"NOT_CAPTURED","rows":rows}

def functional_exploration_gate(summary: dict) -> dict:
    reasons=[]
    if summary.get("interaction_opportunity_steps", 0) > 0 and summary.get("interaction_selected", 0) == 0:
        reasons.append("INTERACTION_OPPORTUNITY_WITH_ZERO_INTERACTION")
    if summary.get("click_opportunity_steps", 0) > 0 and summary.get("click_selected", 0) == 0:
        reasons.append("CLICK_OPPORTUNITY_WITH_ZERO_CLICK")
    if summary.get("functional_state_transitions", 0) == 0:
        reasons.append("NO_FUNCTIONAL_STATE_TRANSITION")
    if summary.get("interaction_opportunity_steps", 0) > 0 and summary.get("interaction_selected", 0) == 0 and summary.get("viewport_only_steps", 0) > 0:
        reasons.append("VIEWPORT_ONLY_POLICY")
    return {"status":"PASS" if not reasons else "FAIL", "failure_reasons":reasons, "performance_ranking_eligible":not reasons, "same_viewport_noop_metric":"NOT_CAPTURED"}

def main():
    split=json.loads((ROOT/"configs/step7_full_site_split.json").read_text(encoding="utf-8")); sites_cfg=json.loads((ROOT/"configs/rl_errorsite_full_training_sites.json").read_text(encoding="utf-8")); sites={str(x["site_id"]):dict(x,cluster="") for x in sites_cfg["sites"]}; clusters=json.loads((ROOT/"DQN/artifacts/training_v3/step7_complete/structural_cluster_manifest.json").read_text(encoding="utf-8"))["clusters"]
    for cluster, ids in clusters.items():
        for sid in ids:
            if sid in sites: sites[sid]["cluster"]=cluster
    val_ids=list(split["validation"])
    diagnostic_limit=int(os.environ.get("STEP7_VALIDATION_LIMIT", "0") or 0)
    if diagnostic_limit > 0:
        val_ids = val_ids[:diagnostic_limit]
    overlap={"train_validation":len(set(split["train"])&set(split["validation"])),"train_test":len(set(split["train"])&set(split["test"])),"validation_test":len(set(split["validation"])&set(split["test"]))}
    all_rows=[]; summaries=[]; sha_before={}
    selector=os.environ.get("STEP7_PPO_VALIDATION_SELECTOR", "sample").strip().lower()
    for arch,seed,prefix in RUNS:
        torch.manual_seed(100000 + int(seed))
        final=OUT/f"{prefix}_full.pt"; sha_before[prefix]=digest(final); payload=torch.load(final,map_location="cpu",weights_only=False); model=RecurrentPPOActorCritic(135,640,128) if arch=="gru" else ObservationV2HistoryMLP(135,640); model.load_state_dict(payload["model_state_dict"]); model.eval(); recurrent=arch=="gru"; rows=[]
        for sid in val_ids:
            print(json.dumps({"event":"VALIDATION_SITE_START","architecture":arch,"seed":seed,"site_id":sid}), flush=True)
            rows.append(evaluate(model,recurrent,sites[sid],15,selector=selector))
            print(json.dumps({"event":"VALIDATION_SITE_COMPLETE","architecture":arch,"seed":seed,"site_id":sid}), flush=True)
        for row in rows: row.update({"architecture":arch,"seed":seed,"checkpoint":str(final),"checkpoint_sha256":sha_before[prefix]})
        metrics=[k for k in rows[0] if k.endswith("coverage") or k.endswith("rate") or k in {"candidate_discovery","exploration_diversity"}]
        summary={k:{"mean":mean([r[k] for r in rows]),"median":median([r[k] for r in rows]),"std":std([r[k] for r in rows])} for k in metrics}; summary.update({"architecture":arch,"seed":seed,"checkpoint":str(final),"checkpoint_sha256":sha_before[prefix],"sites":len(rows),"optimizer_steps":0,"weight_updates":0,"runtime_errors":sum(r["runtime_errors"] for r in rows),"interaction_opportunity_steps":sum(r["interaction_opportunity_steps"] for r in rows),"interaction_mask_valid_steps":sum(r["interaction_mask_valid_steps"] for r in rows),"interaction_selected":sum(sum(1 for x in r["rows"] if x["action_type"] in {"click_element","fill_input","select_option","press_enter"}) for r in rows),"click_opportunity_steps":sum(r["click_opportunity_steps"] for r in rows),"click_selected":sum(r["click_selected"] for r in rows),"fill_opportunity_steps":sum(r["fill_opportunity_steps"] for r in rows),"fill_selected":sum(r["fill_selected"] for r in rows),"select_opportunity_steps":sum(r["select_opportunity_steps"] for r in rows),"select_selected":sum(r["select_selected"] for r in rows),"enter_opportunity_steps":sum(r["enter_opportunity_steps"] for r in rows),"enter_selected":sum(r["enter_selected"] for r in rows),"functional_state_transitions":sum(r["functional_state_transitions"] for r in rows),"viewport_only_steps":sum(r["viewport_only_steps"] for r in rows),"rows":rows}); summary["integrity_gate"]="PASS" if summary["optimizer_steps"]==0 and summary["weight_updates"]==0 and summary["runtime_errors"]==0 else "FAIL"; summary["functional_exploration_gate"]=functional_exploration_gate(summary); summaries.append(summary); all_rows.extend(rows)
    after={p:digest(OUT/f"{p}_full.pt") for _,_,p in RUNS}; unchanged=sha_before==after
    by_seed=[]
    for seed in (42,43,44):
        m=next(x for x in summaries if x["architecture"]=="history_mlp" and x["seed"]==seed); g=next(x for x in summaries if x["architecture"]=="gru" and x["seed"]==seed); by_seed.append({"seed":seed,"mlp":m,"gru":g,"differences":{k:g[k]["mean"]-m[k]["mean"] for k in metrics}})
    aggregate={arch:{k:{"mean":mean([s[k]["mean"] for s in summaries if s["architecture"]==arch]),"median":median([s[k]["mean"] for s in summaries if s["architecture"]==arch]),"std":std([s[k]["mean"] for s in summaries if s["architecture"]==arch])} for k in metrics} for arch in ("history_mlp","gru")}
    result={"schema_version":"rawd_step7_ppo_validation_v1","split":{"train":len(split["train"]),"validation":len(val_ids),"test":len(split["test"]),"validation_clusters":len(split["validation_clusters"]),"overlap":overlap,"test_accessed":False},"protocol":{"episodes_per_site":1,"max_steps":15,"optimizer_steps":0,"weight_updates":0,"backward":False,"selection_method":selector,"selection_seed_base":100000},"per_site":all_rows,"per_run":summaries,"ablation_by_seed":by_seed,"aggregate":aggregate,"checkpoint_sha_unchanged":unchanged,"validation_pass":len(val_ids)==24 and not any(overlap.values()) and unchanged and all(s["integrity_gate"]=="PASS" and s["functional_exploration_gate"]["status"]=="PASS" for s in summaries)}
    (OUT/"ppo_validation_results_canonical.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8"); print(json.dumps({"validation_pass":result["validation_pass"],"runs":len(summaries),"sites_per_run":len(val_ids),"optimizer_steps":0,"weight_updates":0,"sha_unchanged":unchanged},indent=2))
    return 0 if result["validation_pass"] else 2
if __name__=="__main__": raise SystemExit(main())
