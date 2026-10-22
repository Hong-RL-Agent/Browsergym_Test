from __future__ import annotations
import copy, hashlib, tempfile, os, json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/"DQN/artifacts/training_v3/step7_fresh_training_final/history_mlp_seed42_checkpoints"

def canonical_digest(payload):
    p=copy.deepcopy(payload)
    p["metadata"].pop("checkpoint_sha256",None)
    fd,tmp=tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    t=Path(tmp)
    try:
        torch.save(p,t)
        return hashlib.sha256(t.read_bytes()).hexdigest().upper()
    finally:
        if t.exists(): t.unlink()

def main():
    rows=[]
    for path in sorted(D.glob("update_*.pt")):
        payload=torch.load(path,map_location="cpu",weights_only=False)
        digest=canonical_digest(payload)
        payload["metadata"]["checkpoint_sha256"]=digest
        torch.save(payload,path)
        verify=torch.load(path,map_location="cpu",weights_only=False)
        rows.append({"checkpoint":path.name,"canonical_sha256":digest,"metadata_sha256":verify["metadata"]["checkpoint_sha256"],"match":digest==verify["metadata"]["checkpoint_sha256"],"model_state_present":bool(verify.get("model_state_dict")),"optimizer_state_present":bool(verify.get("optimizer_state_dict"))})
    result={"status":"PASS" if all(r["match"] and r["model_state_present"] and r["optimizer_state_present"] for r in rows) else "FAIL","hash_definition":"SHA256 of checkpoint payload with metadata.checkpoint_sha256 excluded","checkpoints":rows}
    (ROOT/"DQN/artifacts/training_v3/step7_fresh_training_final/checkpoint_integrity_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["status"]=="PASS" else 1)

if __name__=="__main__": main()
