"""Parse training logs -> summary json + figures (run: python analyze.py)."""
import ast, glob, json, os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))
LOGS, RES, FIG = f"{D}/logs", f"{D}/results", f"{D}/results/figures"
os.makedirs(FIG, exist_ok=True)

def load(name):
    p = f"{LOGS}/{name}.log"
    if not os.path.exists(p): return None
    txt = open(p).read()
    vals = []
    for m in re.finditer(r"VALIDATION \(Step (\d+)/\d+\).*?STATISTICS.*?\n(.*?)\n={10,}", txt, re.S):
        blk = m.group(0); st = {"step": int(m.group(1))}
        for key, pat in [("sent", r"Mean sentiment:\s+([\d.]+)"), ("std", r"Std:\s+([\d.]+)"),
                         ("pos", r"Positive ratio:\s+([\d.]+)%"), ("len", r"Mean length \(words\):\s+([\d.]+)"),
                         ("ppl", r"Mean perplexity:\s+([\d.]+)"), ("ppl1", r"First-token perplexity:\s+([\d.]+)")]:
            mm = re.search(pat, blk); st[key] = float(mm.group(1)) if mm else None
        ex = re.findall(r"\[\d\] Sentiment: ([\d.]+).*?\n\s+Prompt: (.*?)\n\s+Generated:(.*?)\n", blk, re.S)
        st["examples"] = [{"sent": float(a), "prompt": b, "gen": c.strip()} for a, b, c in ex]
        vals.append(st)
    train = []
    for line in txt.splitlines():
        if line.startswith("{'loss'") and "'reward'" in line:
            try: d = ast.literal_eval(line)
            except Exception: continue
            train.append({"step": 20 * (len(train) + 1), "reward": d["reward"], "reward_std": d["reward_std"],
                          "frac_zero_std": d.get("frac_reward_zero_std"), "len": d.get("completions/mean_length")})
    return {"val": vals, "train": train}

def fin(r): return r["val"][-1] if r and r["val"] else None

if __name__ == "__main__":
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{LOGS}/*.log"))
    allr = {n: load(n) for n in names}
    json.dump(allr, open(f"{RES}/all_runs.json", "w"), indent=1)
    # ---- Ex 2/3
    if allr.get("e2_five_stars") and allr.get("e3_expectation"):
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        for n, lab in (("e2_five_stars", "five_stars (binary)"), ("e3_expectation", "expectation (continuous)")):
            r = allr[n]
            ax[0].plot([t["step"] for t in r["train"]], [t["reward"] for t in r["train"]], marker="o", label=lab)
            ax[1].plot([v["step"] for v in r["val"]], [v["sent"] for v in r["val"]], marker="o", label=lab)
            ax[2].plot([v["step"] for v in r["val"]], [v["ppl"] for v in r["val"]], marker="o", label=lab)
        for a, t in zip(ax, ("train reward (mean over group)", "validation mean sentiment", "validation reference perplexity")):
            a.set_title(t); a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(f"{FIG}/ex2_ex3.png", dpi=130); plt.close()
    # ---- Ex 4
    def sk(n): return float(n.split("_c")[1].split("_")[0])
    fwd = sorted([n for n in names if n.startswith("e4_fwd") and "s400" not in n], key=sk, reverse=True)
    bwd = sorted([n for n in names if n.startswith("e4_bwd") and "s400" not in n], key=sk, reverse=True)
    long_ = [n for n in names if n.startswith("e4_") and "s400" in n]
    if fwd:
        fig, ax = plt.subplots(2, 2, figsize=(13, 8)); ax = ax.ravel()
        for a_, grp, t in ((ax[0], fwd, "forward KL (200 steps)"), (ax[1], bwd, "backward KL (200 steps)")):
            for n in grp:
                r = allr[n]; a_.plot([v["step"] for v in r["val"]], [v["sent"] for v in r["val"]], marker=".", label="coef " + n.split("_c")[1])
            a_.set_title("validation sentiment, " + t); a_.axhline(0.8, c="gray", ls=":"); a_.legend(fontsize=7, ncol=2); a_.grid(alpha=.3); a_.set_xlabel("step")
        for n in long_:
            r = allr[n]
            ax[2].plot([v["step"] for v in r["val"]], [v["sent"] for v in r["val"]], marker=".", label=n[3:])
        ax[2].set_title("validation sentiment, 400-step runs"); ax[2].axhline(0.8, c="gray", ls=":"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3); ax[2].set_xlabel("step")
        for n in fwd + bwd + long_:
            f = fin(allr[n]); ax[3].scatter(f["ppl"], f["sent"], marker="o" if "fwd" in n else "s", c="C2" if "s400" in n else ("C0" if "fwd" in n else "C1"))
            ax[3].annotate(n[3:].replace("_s400", "*"), (f["ppl"], f["sent"]), fontsize=6)
        e3 = fin(allr.get("e3_expectation"))
        if e3: ax[3].scatter(e3["ppl"], e3["sent"], marker="*", s=140, c="k"); ax[3].annotate("no KL", (e3["ppl"], e3["sent"]), fontsize=7)
        ax[3].set_title("final sentiment vs perplexity (blue fwd, orange bwd, green 400 steps)"); ax[3].axhline(0.8, c="gray", ls=":")
        ax[3].set_xlabel("reference perplexity"); ax[3].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{FIG}/ex4.png", dpi=130); plt.close()
    # ---- Ex 5
    k5 = [n for n in names if n.startswith("e5_")]
    if k5:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for n in k5:
            r = allr[n]
            if not r or not r["val"]: continue
            ax[0].plot([v["step"] for v in r["val"]], [v["sent"] for v in r["val"]], marker=".", label=n[3:])
            ax[1].plot([v["step"] for v in r["val"]], [v["len"] for v in r["val"]], marker=".", label=n[3:])
            ax[2].plot([v["step"] for v in r["val"]], [v["ppl"] for v in r["val"]], marker=".", label=n[3:])
        for a, t in zip(ax, ("validation mean sentiment", "mean length (words)", "reference perplexity")):
            a.set_title(t); a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(f"{FIG}/ex5.png", dpi=130); plt.close()
    # ---- Ex 5 bars + Ex 6
    ev = f"{RES}/eval_all.json"
    if os.path.exists(ev):
        E = json.load(open(ev)); ms = ["sentiment", "R_power", "R_norep", "R_short", "R_whiplash", "distinct_frac"]
        order = [n for n in ("base_gpt2", "e4_fwd_c0.1_s400", "e5_power", "e5_norep", "e5_short", "e5_whiplash", "e3_no_kl") if n in E]
        fig, ax = plt.subplots(figsize=(12, 4.5)); w = 0.8 / len(order)
        for i, n in enumerate(order):
            ax.bar([j + i * w for j in range(len(ms))], [E[n][m] for m in ms], w, label=n)
        ax.set_xticks([j + 0.4 for j in range(len(ms))]); ax.set_xticklabels(ms); ax.axhline(0, c="k", lw=.5)
        ax.set_title("every model scored on every metric (192 samples each; columns = metrics)"); ax.legend(fontsize=7, ncol=4); ax.grid(alpha=.3, axis="y")
        plt.tight_layout(); plt.savefig(f"{FIG}/ex5_metrics.png", dpi=130); plt.close()
    pe = {}
    for st in ("norep", "whiplash"):
        for rep in (1, 2):
            p = f"{LOGS}/e6_{st}_rep{rep}.log"
            if not os.path.exists(p): continue
            txt = open(p).read(); sec = txt[txt.rindex("SUMMARY"):]
            for m in re.finditer(r"^  (\S.*?)\s{2,}(-?[\d.]+)\s+(-?[\d.]+)\s*$", sec, re.M):
                pe.setdefault(st, {}).setdefault(m.group(1).strip(), []).append((float(m.group(2)), float(m.group(3))))
    if pe:
        json.dump(pe, open(f"{RES}/ex6.json", "w"), indent=1)
        fig, ax = plt.subplots(1, len(pe), figsize=(6.5 * len(pe), 4))
        ax = [ax] if len(pe) == 1 else ax
        for a_, (st, d) in zip(ax, pe.items()):
            names_ = list(d); import numpy as np
            b = [np.mean([x[0] for x in d[n]]) for n in names_]; t = [np.mean([x[1] for x in d[n]]) for n in names_]
            xs = np.arange(len(names_)); a_.bar(xs - .2, b, .4, label="base GPT-2"); a_.bar(xs + .2, t, .4, label="RL-trained")
            a_.set_xticks(xs); a_.set_xticklabels(names_, rotation=25, ha="right", fontsize=8); a_.set_title(f"Ex 6, metric: {st} (mean of 2 repeats)"); a_.legend(); a_.grid(alpha=.3, axis="y")
        plt.tight_layout(); plt.savefig(f"{FIG}/ex6.png", dpi=130); plt.close()
    print("ok", len(allr), "runs")
