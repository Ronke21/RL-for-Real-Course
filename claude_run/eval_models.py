"""Post-hoc evaluation: generate with each model (validation settings) and score with every metric.
usage: python eval_models.py OUT.json name=path [name=path ...]"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
import rewards
from data import VALIDATION_PROMPTS
from evaluate import load_model

out, specs = sys.argv[1], sys.argv[2:]
res = {}
for spec in specs:
    name, path = spec.split("=", 1)
    torch.manual_seed(0)
    model, tok = load_model(path)
    tok.padding_side = "left"
    gens = []
    with torch.no_grad():
        for rep in range(3):
            for i in range(0, len(VALIDATION_PROMPTS), 16):
                bp = list(VALIDATION_PROMPTS[i:i+16])
                x = tok(bp, return_tensors="pt", padding=True).to(model.device)
                o = model.generate(**x, max_new_tokens=64, temperature=1.5, top_p=0.9, do_sample=True,
                                   pad_token_id=tok.pad_token_id)
                for p, seq in zip(bp, o):
                    t = tok.decode(seq, skip_special_tokens=True)
                    gens.append((p, t[len(p):] if t.startswith(p) else t))
    texts = [g for _, g in gens]
    sent = rewards.expected_stars(texts)
    m = {"n": len(texts), "sentiment": float(np.mean(sent)), "positive_ratio": float(np.mean(np.array(sent) > 0.5)),
         "words": float(np.mean([len(t.split()) for t in texts])),
         "distinct_frac": float(np.mean([len(set(w.lower() for w in t.split())) / max(len(t.split()), 1) for t in texts]))}
    for mode in ("power", "short", "norep", "whiplash"):
        os.environ["SHAPE_MODE"] = mode
        m["R_" + mode] = float(np.mean(rewards.shaped_reward(sent, texts)))
    m["samples"] = [{"prompt": p, "gen": g[:220]} for p, g in gens[:6]]
    res[name] = m
    print(name, {k: round(v, 3) for k, v in m.items() if k != "samples"}, flush=True)
    del model; torch.cuda.empty_cache()
    json.dump(res, open(out, "w"), indent=1)
