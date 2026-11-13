"""Why the FN chirps with a detected harmonic are (not) recovered (dual-head, 0.5)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import load_sim, at, kept, evaluate_at, PROD, OFF, _f_at, OUT  # noqa: E402

t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
det = sys.argv[1] if len(sys.argv) > 1 else "dual-head"
preds, truth, tracks = load_sim(det)
labels = truth.labels.reset_index(drop=True)
prod = at(preds, t, tracks, PROD)
off = at(preds, t, tracks, OFF if "emitter_hz" in preds else PROD)
_, _, l_off = evaluate_at(kept(off), truth, t)
_, m_prod, l_prod = evaluate_at(kept(prod), truth, t)
harm = labels[(labels["label"] == "ignore") & (labels["reason"].astype(str) == "harmonic")]
hkey = {(r, c): i for i, r, c in zip(harm.index, harm["recording"], harm["chirp_id"], strict=True)}
rows = []
for li in labels.index[(labels["label"] == "chirp").to_numpy()]:
    if l_off.get(li) != "fn":
        continue
    lab = labels.loc[li]
    hi = hkey.get((lab["recording"], lab["chirp_id"]))
    if hi is None:
        continue
    h = labels.loc[hi]
    g = prod[prod["recording"] == lab["recording"]]
    gc = 0.5 * (g["t_start_s"] + g["t_end_s"])
    hit = (np.abs(gc - h["t_center_s"]) <= 0.1) & (g["f_low_hz"] < h["f_high_hz"]) & (g["f_high_hz"] > h["f_low_hz"])
    if not hit.any():
        continue
    c = g[hit].sort_values("score").iloc[-1]
    k = int(lab["emitter_id"])
    f_k = _f_at(tracks[lab["recording"]], k, lab["t_center_s"])
    kp = kept(g)
    kc = 0.5 * (kp["t_start_s"] + kp["t_end_s"])
    near = kp[(np.abs(kc - lab["t_center_s"]) <= 0.3) & (kp["f_low_hz"] <= f_k) & (kp["f_high_hz"] >= f_k)]
    rows.append({
        "recording": lab["recording"], "t_label": round(lab["t_center_s"], 3), "fish": k, "f_k": round(f_k, 1),
        "label_f": f"{lab['f_low_hz']:.0f}-{lab['f_high_hz']:.0f}",
        "copy_score": round(c["score"], 2), "copy_reason": c["drop_reason"] or ("reattributed" if c["reattributed"] == "harmonic" else "kept"),
        "copy_emitter": round(c.get("emitter_hz", np.nan), 1), "copy_harmonic_of": c["harmonic_of_id"],
        "kept_spanning_f_k": "; ".join(f"{r.score:.2f}@{0.5*(r.t_start_s+r.t_end_s)-lab['t_center_s']:+.2f}s {r.f_low_hz:.0f}-{r.f_high_hz:.0f} id {r.emitter_id if 'emitter_id' in kp else ''}" for r in near.itertuples()),
        "recovered": l_prod.get(li) == "tp",
    })
out = pd.DataFrame(rows)
out.to_csv(OUT / "tables" / f"upper_bound_cases_{det}_{t}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(out.to_string(index=False))
