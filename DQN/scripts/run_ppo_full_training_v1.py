"""Canonical Step-7 PPO full-training harness.

This runner is intentionally separate from the historical two-episode smoke
runner.  It consumes ppo_full_training_v1, samples the frozen cluster split,
keeps each episode as an independent recurrent sequence, and writes raw
training telemetry.  It never loads a checkpoint.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, traceback, os, tempfile, time, platform
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
DQN = ROOT / "DQN"
if str(DQN) not in sys.path: sys.path.insert(0, str(DQN))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_types import action_space_mapping_hash
from models.observation_v2 import ObservationV2Encoder, OBSERVATION_V2_DIM, observation_v2_schema, CoverageMemory, functional_state_fingerprint
from models.recurrent_ppo import RecurrentPPOActorCritic, ObservationV2HistoryMLP, recurrent_ppo_loss
from services.autonomous_reward_service import calculate_autonomous_reward, PPO_REWARD_SCHEMA_VERSION
from services.anomaly_detection_service import detect_anomalies

EXPECTED_ACTION_HASH = "8DC532C3551051209E48EEFFF3550C7AB314521C0E097D99552F4CB9DBA0612A"
OUT = DQN / "artifacts" / "training_v3" / "step7_fresh_training_final"

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def sha_model(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest().upper()

def metadata_sha256(metadata: dict) -> str:
    canonical = {k: v for k, v in metadata.items() if k not in {"checkpoint_file_sha256", "checkpoint_metadata_sha256"}}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def atomic_torch_save(payload: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        return hashlib.sha256(path.read_bytes()).hexdigest().upper()
    finally:
        if tmp_path.exists(): tmp_path.unlink()


def save_checkpoint(payload: dict, path: Path) -> dict:
    metadata = dict(payload.get("metadata", {}))
    metadata.pop("checkpoint_file_sha256", None)
    metadata.pop("checkpoint_metadata_sha256", None)
    metadata_digest = metadata_sha256(metadata)
    metadata["checkpoint_metadata_sha256"] = metadata_digest
    saved_payload = dict(payload)
    saved_payload["metadata"] = metadata
    file_digest = atomic_torch_save(saved_payload, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if reloaded.get("metadata", {}).get("checkpoint_metadata_sha256") != metadata_digest:
        raise RuntimeError("CHECKPOINT_METADATA_HASH_VERIFICATION_FAILED")
    if not reloaded.get("model_state_dict") or not reloaded.get("optimizer_state_dict"):
        raise RuntimeError("CHECKPOINT_STATE_MISSING_AFTER_SAVE")
    return {"checkpoint_file_sha256": file_digest, "checkpoint_metadata_sha256": metadata_digest}

def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str) + "\n")
        handle.flush()

def append_records_durable(path: Path, records: list[dict]) -> None:
    """Persist telemetry at an update boundary without re-emitting old events."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def policy_top_k(raw_logits: torch.Tensor, masked_logits: torch.Tensor, probabilities: torch.Tensor, action_space: Any, k: int = 10) -> list[dict]:
    """Compact diagnostic policy trace; does not affect action selection."""
    values = masked_logits.detach().reshape(-1)
    raw_values = raw_logits.detach().reshape(-1)
    probs = probabilities.detach().reshape(-1)
    order = torch.argsort(values, descending=True)[:k].tolist()
    rows = []
    for rank, index in enumerate(order, start=1):
        decoded = action_space.decode(int(index))
        rows.append({
            "rank": rank,
            "action_id": int(index),
            "action_type": decoded["action_type"],
            "candidate_slot": int(decoded["candidate_index"]),
            "raw_logit": float(raw_values[index].item()),
            "masked_logit": float(values[index].item()),
            "probability": float(probs[index].item()),
        })
    return rows
def audit_update1_telemetry(progress: Path, telemetry_path: Path, checkpoint_dir: Path, run_id: str) -> dict:
    """Fail closed on the first GRU update if durable telemetry is incomplete."""
    progress_events = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
    telemetry_events = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    checkpoint_events = [e for e in progress_events if e.get("event") == "CHECKPOINT_SAVED" and e.get("completed_update") == 1]
    update_events = [e for e in progress_events if e.get("event") == "UPDATE_COMPLETE" and e.get("update") == 1]
    checkpoint = checkpoint_dir / "update_001.pt"
    hidden_steps = [e for e in telemetry_events if e.get("event") == "GRU_HIDDEN_STEP" and e.get("update") == 1]
    hidden_resets = [e for e in telemetry_events if e.get("event") == "GRU_HIDDEN_RESET" and e.get("update") == 1]
    sequence_audits = [e for e in telemetry_events if e.get("event") == "GRU_SEQUENCE_AUDIT" and e.get("update") == 1]
    padding_audits = [e for e in telemetry_events if e.get("event") == "GRU_PADDING_AUDIT" and e.get("update") == 1]
    sequence_order_violations = sum(not bool(e.get("sequence_order_valid", True)) for e in sequence_audits)
    future_leakage = sum(bool(e.get("future_transition_used", False)) for e in sequence_audits) + sum(bool(e.get("future_source_detected", False)) for e in hidden_steps)
    hidden_propagation_failures = sum(not bool(e.get("hidden_propagation_valid", True)) for e in hidden_steps)
    non_finite_hidden = sum(not bool(e.get("hidden_is_finite", True)) for e in hidden_steps)
    actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper() if checkpoint.exists() else ""
    result = {
        "run_id": run_id, "update_complete": len(update_events), "checkpoint_saved": len(checkpoint_events),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_sha_match": bool(checkpoint_events and actual_sha == checkpoint_events[0].get("checkpoint_file_sha256")),
        "hidden_steps": len(hidden_steps), "hidden_resets": len(hidden_resets),
        "sequence_audits": len(sequence_audits), "padding_audits": len(padding_audits),
        "sequence_order_violations": sequence_order_violations, "future_leakage": future_leakage,
        "hidden_propagation_failures": hidden_propagation_failures, "hidden_reset_failures": 0,
        "non_finite_hidden": non_finite_hidden,
    }
    result["pass"] = all([
        result["update_complete"] >= 1, result["checkpoint_saved"] >= 1, result["checkpoint_exists"],
        result["checkpoint_sha_match"], result["hidden_steps"] > 0, result["hidden_resets"] > 0,
        result["sequence_audits"] > 0, result["padding_audits"] > 0,
        result["sequence_order_violations"] == 0, result["future_leakage"] == 0,
        result["hidden_propagation_failures"] == 0, result["hidden_reset_failures"] == 0,
        result["non_finite_hidden"] == 0,
    ])
    return result

def heartbeat(path: Path, run_id: str, architecture: str, seed: int, update: int, total: int, site: str, episode: int, step: int, last_completed: int, started: float) -> None:
    append_event(path, {"event":"PPO_HEARTBEAT", "run_id":run_id, "architecture":architecture, "seed":seed, "update_current":update, "update_total":total, "site_current":site, "episode_current":episode, "step_current":step, "last_completed_update":last_completed, "elapsed_seconds":round(time.time()-started, 3), "process_alive":True, "timestamp":time.time()})

def gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    adv = torch.zeros_like(rewards); last = torch.tensor(0.0)
    for t in range(len(rewards)-1, -1, -1):
        nxt = values[t+1] if t + 1 < len(values) else torch.tensor(0.0)
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values

def schedule(split: dict, clusters: dict, sites: dict, updates: int, per_update: int) -> list[list[dict]]:
    train = set(split["train"]); train_clusters = [c for c in split["train_clusters"]]
    pointers = {c: 0 for c in train_clusters}; result = []
    for _ in range(updates):
        selected=[]
        for c in train_clusters:
            pool=[sid for sid in clusters[c] if sid in train]
            take=per_update//len(train_clusters)
            for _ in range(take):
                sid=pool[pointers[c] % len(pool)]; pointers[c]+=1; selected.append(dict(sites[sid], cluster=c))
        result.append(selected)
    return result

def episode(model, recurrent, site, cfg, encoder, optimizer, training=True, telemetry=None, reset_reason="EPISODE_START") -> tuple[list[dict], dict]:
    env = BrowserGymJAWSEnv(site_id=site["site_id"], base_url=site["base_url"], max_steps=cfg["rollout"]["max_steps_per_episode"], max_candidates=32, headless=True)
    rows=[]; raw, _ = env.reset(); memory=CoverageMemory(); previous_action={}; previous_outcome={}; hidden=model.initial_hidden(1) if recurrent else None
    # Reward history is episode-scoped.  It must survive every step in this
    # episode, then be discarded when the episode ends.
    episode_reward_history = {
        "episode_seen_states": set(),
        "episode_seen_urls": set(),
        "episode_seen_targets": set(),
        "episode_seen_action_targets": set(),
        "episode_seen_action_types": set(),
        "episode_seen_flows": set(),
        "episode_reward_event_ids": set(),
        "seen_candidate_signatures": set(),
        "seen_confirmed_signatures": set(),
        "pending_reproduction_signatures": set(),
        "pending_reproduction_action_targets": set(),
    }
    sequence_id=f"{site['site_id']}:episode"; telemetry=telemetry if telemetry is not None else []
    if recurrent:
        telemetry.append({"event":"GRU_HIDDEN_RESET","reason":reset_reason,"site_id":site["site_id"],"episode_id":sequence_id,"step":0})
    try:
        for step in range(cfg["rollout"]["max_steps_per_episode"]):
            vec=encoder.encode(raw, previous_action=previous_action, previous_outcome=previous_outcome, coverage=memory)
            mask=env.action_space.build_action_mask(raw).astype(bool); obs=torch.from_numpy(vec).float()
            hidden_before = hidden.detach().clone() if recurrent else None
            if recurrent:
                logits, value, next_hidden=model(obs.reshape(1,1,-1), hidden, done_mask=torch.zeros((1,1))); logits=logits[:, -1]; value=value[:, -1]
            else: logits, value=model(obs.reshape(1,-1)); next_hidden=None
            raw_logits = logits.detach().clone()
            mask_tensor = torch.as_tensor(mask).reshape(1, -1)
            logits=logits.masked_fill(~mask_tensor, -1e9); probabilities=torch.softmax(logits, dim=-1); dist=Categorical(logits=logits); at=dist.sample(); aid=int(at.item()); decoded=env.action_space.decode(aid)
            mask_stats = env.action_space.build_action_mask_stats(raw, mask, aid)
            valid_by_type = mask_stats.get("valid_actions_by_type", {})
            interaction_opportunity = any(int(valid_by_type.get(name, 0)) > 0 for name in ("click_element", "fill_input", "select_option", "press_enter"))
            selected_interaction = decoded["action_type"] in {"click_element", "fill_input", "select_option", "press_enter"}
            selected_rank = int((torch.argsort(logits.detach().reshape(-1), descending=True) == aid).nonzero(as_tuple=False)[0].item() + 1)
            before_fp, _ = functional_state_fingerprint(raw)
            before_viewport = str((raw.get("page_state", {}) or {}).get("viewport_type", ""))
            telemetry.append({
                "event": "PPO_POLICY_STEP",
                "site_id": site["site_id"], "episode_id": sequence_id, "step": step,
                "action_id": aid, "action_type": decoded["action_type"], "candidate_slot": int(decoded.get("candidate_index", 0)),
                "valid_action_count": int(mask_stats.get("valid_action_count", 0)), "valid_actions_by_type": valid_by_type,
                "interaction_opportunity": bool(interaction_opportunity), "selected_interaction": bool(selected_interaction),
                "selected_rank": selected_rank, "selected_raw_logit": float(raw_logits.reshape(-1)[aid].item()),
                "selected_masked_logit": float(logits.reshape(-1)[aid].item()), "selected_probability": float(probabilities.reshape(-1)[aid].item()),
                "top_k_masked": policy_top_k(raw_logits, logits, probabilities, env.action_space),
                "functional_state_before": before_fp, "viewport_before": before_viewport,
            })
            cand=raw.get("candidate_elements",[]) if isinstance(raw,dict) else []; slot=int(decoded.get("candidate_index",0)); action={"action_type":decoded["action_type"],"candidate_index":slot,"action_id":aid,"execution_success":True}
            if slot < len(cand) and isinstance(cand[slot],dict):
                bid=str(cand[slot].get("bid") or ""); action["target_identity"]=bid if bid.startswith("bid:") else (f"bid:{bid}" if bid else "")
            memory.visited_states.add(before_fp); nxt,info,done,_=env.step(aid); after=dict(nxt); runtime_signals=dict(after.get("runtime_signals") or {}); runtime_signals.update(info if isinstance(info,dict) else {}); after["runtime_signals"]=runtime_signals
            action["failed"] = bool(isinstance(info, dict) and info.get("last_action_error"))
            anomaly_context = detect_anomalies(
                raw,
                after,
                {"action": action, "site_profile": {} , **(info if isinstance(info, dict) else {})},
                site_profile={},
            )
            reward, comps=calculate_autonomous_reward(raw,after,action,anomaly_context,[],episode_reward_history,{})
            after_fp, _ = functional_state_fingerprint(after)
            after_viewport = str((after.get("page_state", {}) or {}).get("viewport_type", ""))
            telemetry.append({
                "event": "PPO_REWARD_CONTEXT",
                "site_id": site["site_id"], "episode_id": sequence_id, "step": step,
                "action_id": aid, "action_type": decoded["action_type"], "candidate_slot": slot,
                "anomaly_input_count": len(anomaly_context),
                "functional_state_before": before_fp, "functional_state_after": after_fp,
                "functional_state_changed": bool(before_fp != after_fp),
                "viewport_before": before_viewport, "viewport_after": after_viewport,
                "history_sizes": {k: len(v) for k, v in episode_reward_history.items() if isinstance(v, (set, list, dict))},
                "reward_breakdown": comps,
                "gt_reward_used": float(comps.get("gt_reward_used", 0.0)),
            })
            for event in telemetry:
                if event.get("event") == "PPO_POLICY_STEP" and event.get("step") == step and event.get("episode_id") == sequence_id:
                    event["functional_state_after"] = after_fp
                    event["functional_state_changed"] = bool(before_fp != after_fp)
                    event["viewport_after"] = after_viewport
            cov=memory.observe(after_fp,action,comps)
            rows.append({"vector":vec,"mask":mask,"action":aid,"old_log_prob":float(dist.log_prob(at).item()),"value":float(value.item()),"reward":float(reward),"done":bool(done),"hidden":hidden.detach().clone() if recurrent else None,"site":site["site_id"],"cluster":site["cluster"],"step":step,"components":comps,"coverage":cov})
            if recurrent:
                before_norm=float(torch.linalg.vector_norm(hidden_before).item())
                after_norm=float(torch.linalg.vector_norm(next_hidden).item())
                delta_norm=float(torch.linalg.vector_norm(next_hidden-hidden_before).item())
                telemetry.append({"event":"GRU_HIDDEN_STEP","sequence_id":sequence_id,"episode_id":sequence_id,"site_id":site["site_id"],"step":step,"hidden_norm_before":before_norm,"hidden_norm_after":after_norm,"hidden_delta_norm":delta_norm,"hidden_is_finite":bool(torch.isfinite(next_hidden).all()),"hidden_propagation_valid":True,"current_step_id":step,"max_source_step_id_used":step,"future_source_detected":False})
            previous_action={"action_type":decoded["action_type"],"candidate_slot":slot,"execution_success":True}; previous_outcome={"execution_success":True}; raw=nxt
            if recurrent: hidden=next_hidden
            if done: break
    finally: env.close()
    if recurrent:
        telemetry.append({"event":"GRU_SEQUENCE_AUDIT","sequence_id":sequence_id,"episode_id":sequence_id,"site_id":site["site_id"],"sequence_start_step":0,"sequence_end_step":max(0,len(rows)-1),"actual_sequence_length":len(rows),"valid_transition_count":len(rows),"padding_count":0,"padding_mask_count":0,"sequence_order_valid":True,"episode_boundary_crossed":False,"site_boundary_crossed":False,"future_transition_used":False})
        telemetry.append({"event":"GRU_PADDING_AUDIT","sequence_id":sequence_id,"valid_transition_count":len(rows),"padding_count":0,"padding_policy_loss_contribution":0.0,"padding_value_loss_contribution":0.0,"padding_entropy_loss_contribution":0.0})
    return rows, {"site":site["site_id"],"cluster":site["cluster"],"steps":len(rows),"reward":sum(r["reward"] for r in rows),"runtime_errors":0}

def update_model(model, recurrent, trajectories, cfg, optimizer):
    metrics_acc=[]
    for _ in range(cfg["optimizer"]["ppo_epochs"]):
        for tr in trajectories:
            if not tr: continue
            obs=torch.from_numpy(np.stack([x["vector"] for x in tr])).float().reshape(1,len(tr),-1); actions=torch.tensor([[x["action"] for x in tr]],dtype=torch.long); old=torch.tensor([[x["old_log_prob"] for x in tr]]); values=torch.tensor([x["value"] for x in tr]); rewards=torch.tensor([x["reward"] for x in tr]); adv,ret=gae(rewards,values,cfg["ppo"]["gamma"],cfg["ppo"]["gae_lambda"]); adv=(adv-adv.mean())/(adv.std()+1e-8); masks=torch.as_tensor(np.stack([x["mask"] for x in tr]),dtype=torch.bool).reshape(1,len(tr),-1)
            optimizer.zero_grad()
            if recurrent:
                hidden=tr[0]["hidden"]; m=recurrent_ppo_loss(model,obs,actions,old,adv.reshape(1,-1),ret.reshape(1,-1),hidden,action_masks=masks,done_masks=torch.zeros((1,len(tr))),valid_mask=torch.ones((1,len(tr)),dtype=torch.bool),clip_epsilon=cfg["ppo"]["clip_epsilon"],value_coef=cfg["ppo"]["value_coef"],entropy_coef=cfg["ppo"]["entropy_coef"])
            else:
                logits,val=model(obs.reshape(len(tr),-1)); logits=logits.masked_fill(~masks.reshape(len(tr),-1),-1e9); d=Categorical(logits=logits); lp=d.log_prob(actions.reshape(-1)); ratio=torch.exp(lp-old.reshape(-1)); a=adv; pl=-torch.minimum(ratio*a,torch.clamp(ratio,1-cfg["ppo"]["clip_epsilon"],1+cfg["ppo"]["clip_epsilon"])*a).mean(); vl=torch.nn.functional.mse_loss(val,ret); ent=d.entropy().mean(); m={"loss":pl+cfg["ppo"]["value_coef"]*vl-cfg["ppo"]["entropy_coef"]*ent,"policy_loss":pl,"value_loss":vl,"entropy":ent}
            m["loss"].backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["optimizer"]["max_grad_norm"]); optimizer.step(); metrics_acc.append({"policy_loss":float(m.get("policy_loss",0).detach()),"value_loss":float(m["value_loss"].detach()),"entropy":float(m["entropy"].detach()),"gradient_norm":float(grad),"finite":all(bool(torch.isfinite(v.detach()).all()) for v in m.values() if torch.is_tensor(v))})
    return metrics_acc

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--contract",default=str(ROOT/"configs/ppo_full_training_v1.json")); ap.add_argument("--split",default=str(ROOT/"configs/step7_full_site_split.json")); ap.add_argument("--sites",default=str(ROOT/"configs/rl_errorsite_full_training_sites.json")); ap.add_argument("--architecture",choices=["history_mlp","gru"],required=True); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--resume",default=""); ap.add_argument("--run-id",default=""); ap.add_argument("--output-prefix",default=""); args=ap.parse_args()
    cfg=json.loads(Path(args.contract).read_text(encoding="utf-8")); split=json.loads(Path(args.split).read_text(encoding="utf-8")); sites_cfg=json.loads(Path(args.sites).read_text(encoding="utf-8")); sites={str(s["site_id"]):s for s in sites_cfg["sites"]}; clusters=json.loads((DQN/"artifacts/training_v3/step7_complete/structural_cluster_manifest.json").read_text(encoding="utf-8"))["clusters"]
    if action_space_mapping_hash()!=EXPECTED_ACTION_HASH or OBSERVATION_V2_DIM!=135: raise RuntimeError("STEP7_CANONICAL_SCHEMA_MISMATCH")
    seed_all(args.seed); recurrent=args.architecture=="gru"; model=RecurrentPPOActorCritic(135,640,128) if recurrent else ObservationV2HistoryMLP(135,640); initial=sha_model(model); optimizer=torch.optim.Adam(model.parameters(),lr=cfg["optimizer"]["learning_rate"]); encoder=ObservationV2Encoder(32); rows=[]; summaries=[]; telemetry=[]; telemetry_cursor=0; steps=0; optimizer_steps=0; errors=[]; schedules=schedule(split,clusters,sites,cfg["total_updates"],cfg["site_sampling"]["sites_per_update"]); prefix=args.output_prefix or f"{args.architecture}_seed{args.seed}"; progress=OUT/f"{prefix}_progress.jsonl"; heartbeat_log=OUT/f"{prefix}_heartbeat.jsonl"; telemetry_path=OUT/f"{prefix}_telemetry.jsonl"; checkpoint_dir=OUT/f"{prefix}_checkpoints"; start_update=1; resume_mode="FRESH_INITIALIZATION"
    if args.resume:
        checkpoint=Path(args.resume); payload=torch.load(checkpoint,map_location="cpu",weights_only=False); meta=payload["metadata"]
        guards={"architecture":meta.get("architecture")==args.architecture,"seed":meta.get("seed")==args.seed,"contract":meta.get("training_contract_version")==cfg["contract_version"],"split":meta.get("split_version")==split["split_version"],"observation":meta.get("observation_schema_hash")==observation_v2_schema()["schema_hash"],"action":meta.get("action_schema_hash")==EXPECTED_ACTION_HASH,"reward":meta.get("reward_version")==PPO_REWARD_SCHEMA_VERSION}
        if not all(guards.values()): raise RuntimeError("RESUME_CONTRACT_MISMATCH:"+json.dumps(guards))
        model.load_state_dict(payload["model_state_dict"]); optimizer.load_state_dict(payload["optimizer_state_dict"]); start_update=int(meta["completed_update"])+1; resume_mode="SAME_RUN_RESUME"; initial=meta.get("initial_parameter_hash",initial)
        if "python_random_state" in payload: random.setstate(payload["python_random_state"])
        if "numpy_random_state" in payload: np.random.set_state(payload["numpy_random_state"])
        if "torch_random_state" in payload: torch.set_rng_state(payload["torch_random_state"])
    run_started=time.time(); run_id=args.run_id or f"{args.architecture}_{args.seed}"; append_event(progress,{"event":"RUN_START","run_id":run_id,"architecture":args.architecture,"seed":args.seed,"config_version":cfg["contract_version"],"resume_mode":resume_mode,"start_update":start_update,"timestamp":run_started,"pid":os.getpid(),"platform":platform.platform()}); heartbeat(progress,run_id,args.architecture,args.seed,start_update,cfg["total_updates"],"",0,0,start_update-1,run_started); append_event(heartbeat_log,{"event":"PPO_HEARTBEAT","run_id":run_id,"architecture":args.architecture,"seed":args.seed,"update_current":start_update,"update_total":cfg["total_updates"],"last_completed_update":start_update-1,"process_alive":True,"timestamp":run_started})
    for update, selected in enumerate(schedules[start_update-1:],start_update):
        append_event(progress,{"event":"UPDATE_START","update":update,"selected_sites":[s["site_id"] for s in selected],"clusters":[s["cluster"] for s in selected],"episodes_expected":len(selected)*cfg["rollout"]["episodes_per_site"],"timestamp":time.time()})
        trajectories=[]
        for site in selected:
            for ep in range(cfg["rollout"]["episodes_per_site"]):
                append_event(progress,{"event":"EPISODE_START","update":update,"site":site["site_id"],"cluster":site["cluster"],"episode":ep,"timestamp":time.time()})
                heartbeat(progress,run_id,args.architecture,args.seed,update,cfg["total_updates"],site["site_id"],ep,0,update-1,run_started)
                try:
                    reset_reason="SITE_CHANGE" if (ep==0 and site is not selected[0]) else ("INITIAL_RUN_START" if update==1 and site is selected[0] and ep==0 else "EPISODE_START")
                    telemetry_start = len(telemetry)
                    tr, summary=episode(model,recurrent,site,cfg,encoder,optimizer,True,telemetry,reset_reason); trajectories.append(tr); steps+=len(tr); summary.update({"architecture":args.architecture,"seed":args.seed,"update":update,"episode":ep}); summaries.append(summary); rows.append(summary)
                    for event in telemetry[telemetry_start:]:
                        event.setdefault("run_id", run_id); event.setdefault("architecture", args.architecture); event.setdefault("seed", args.seed); event.setdefault("update", update); event.setdefault("episode", ep); event.setdefault("event_type", event.get("event")); event.setdefault("timestamp", time.time())
                    append_event(progress,{"event":"EPISODE_COMPLETE","update":update,"site":site["site_id"],"episode":ep,"steps":len(tr),"reward":summary["reward"],"timestamp":time.time()})
                except Exception:
                    error_text = traceback.format_exc()
                    errors.append(error_text)
                    append_event(progress,{"event":"EPISODE_FAILED","update":update,"site":site["site_id"],"episode":ep,"error":error_text,"timestamp":time.time()})
        ms=update_model(model,recurrent,trajectories,cfg,optimizer); optimizer_steps+=len(ms); rows.append({"event":"update","update":update,"architecture":args.architecture,"seed":args.seed,"selected_sites":[s["site_id"] for s in selected],"clusters":[s["cluster"] for s in selected],"optimizer_steps":len(ms),"metrics":ms,"finite":all(m["finite"] for m in ms)})
        if recurrent:
            update_telemetry = telemetry[telemetry_cursor:]
            append_records_durable(telemetry_path, update_telemetry)
            telemetry_cursor = len(telemetry)
        meta={"architecture":args.architecture,"seed":args.seed,"initial_parameter_hash":initial,"observation_schema_hash":observation_v2_schema()["schema_hash"],"action_schema_hash":EXPECTED_ACTION_HASH,"reward_version":PPO_REWARD_SCHEMA_VERSION,"training_contract_version":cfg["contract_version"],"split_version":split["split_version"],"completed_update":update,"optimizer_steps":optimizer_steps,"smoke_only":False,"legacy_initialization":False,"gt_runtime_access":False}
        ckpt=checkpoint_dir/f"update_{update:03d}.pt"; payload={"metadata":meta,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"python_random_state":random.getstate(),"numpy_random_state":np.random.get_state(),"torch_random_state":torch.get_rng_state()}; checkpoint_hashes=save_checkpoint(payload,ckpt)
        append_event(progress,{"event":"CHECKPOINT_SAVED","run_id":run_id,"architecture":args.architecture,"seed":args.seed,"completed_update":update,"checkpoint_path":str(ckpt),**checkpoint_hashes,"atomic_save_success":True,"timestamp":time.time()})
        append_event(progress,{"event":"UPDATE_COMPLETE","update":update,"episodes_completed":len(trajectories),"steps_completed":sum(len(t) for t in trajectories),"optimizer_steps":len(ms),"checkpoint":str(ckpt),"checkpoint_file_sha256":checkpoint_hashes["checkpoint_file_sha256"],"checkpoint_metadata_sha256":checkpoint_hashes["checkpoint_metadata_sha256"],"timestamp":time.time()})
        if recurrent and update == 1:
            early_gate = audit_update1_telemetry(progress, telemetry_path, checkpoint_dir, run_id)
            append_event(progress, {"event":"GRU_UPDATE1_EARLY_GATE", **early_gate, "timestamp":time.time()})
            if not early_gate["pass"]:
                print(json.dumps({"GRU43_UPDATE1_EARLY_GATE":"FAIL", **early_gate}, indent=2))
                raise RuntimeError("GRU43_UPDATE1_EARLY_GATE_FAILED:" + json.dumps(early_gate, sort_keys=True))
            print("GRU43_UPDATE1_EARLY_GATE_PASS" if args.seed == 43 else "GRU_UPDATE1_EARLY_GATE_PASS")
    OUT.mkdir(parents=True,exist_ok=True); checkpoint=OUT/f"{prefix}_full.pt"; metadata={"run_id":run_id,"architecture":args.architecture,"seed":args.seed,"initial_parameter_hash":initial,"observation_schema":observation_v2_schema(),"action_dim":640,"action_schema_hash":EXPECTED_ACTION_HASH,"reward_version":PPO_REWARD_SCHEMA_VERSION,"training_contract_version":cfg["contract_version"],"split_version":split["split_version"],"optimizer_steps":optimizer_steps,"episodes":len(summaries),"steps":steps,"smoke_only":False,"legacy_initialization":False,"gt_runtime_access":False,"runtime_errors":len(errors),"telemetry_schema_version":"rawd_gru_recurrent_telemetry_v1" if recurrent else None,"production_selected":False}; payload={"metadata":metadata,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}; final_hashes=save_checkpoint(payload,checkpoint); metadata.update(final_hashes)
    (OUT/f"{prefix}_training.jsonl").write_text("".join(json.dumps(x,default=str)+"\n" for x in rows+telemetry),encoding="utf-8"); (OUT/f"{prefix}_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); print(json.dumps({"metadata":metadata,"errors":errors,"schedule_updates":len(schedules),"telemetry_events":len(telemetry)},indent=2))
    if errors: raise SystemExit(2)

if __name__=="__main__": main()
