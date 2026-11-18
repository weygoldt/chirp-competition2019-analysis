"""The numbers of the hazard analysis that the report or manuscript may cite.

New in the paper port (not in chirp_competition b65108e). ``chase_hazard_stats``
collects them from the tables the pipeline wrote into its output directory
(the same rows ``summary.txt`` quotes), and ``chirp_chase_hazard.py`` passes
them to ``manuscript_stats.write_stats("chase_hazard", ...)``. Ratios are
hazard ratios (FE, RE, simulations) or odds ratios (GLMM) with their 95 %
interval; NaN entries are left out (write_stats accepts finite numbers only).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import hz_hazard  # noqa: E402
import hz_sim  # noqa: E402
from hz_config import Config  # noqa: E402
from hz_data import Trial  # noqa: E402

PRIMARY, CHECK, VB, META = "mixed_laplace", "glm_fe", "mixed_vb", "re_meta"
TAU_MODEL, TAU_TERM = "tau_1000ms", "sub_any_1000ms"
GUARD_MODEL, GUARD_TERM = "guard_500ms", "sub_grd_500ms"

# acausal simulation statistics worth citing (hz_acausal_sim stat names)
ACSIM_OFFSET_STATS = ("hr_primary", "ratio_pf", "ratio_grd", "ratio_k", "ratio_rm",
                      "ratio_rm_grd", "local_ratio_1s")
ACSIM_ONSET_STATS = ("on_hr_primary", "on_ratio_pf", "on_ratio_grd")


def _read(f: Path) -> pd.DataFrame:
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f, keep_default_na=False, na_values=[""], float_precision="round_trip")


def _pick(df: pd.DataFrame, **kw: Any) -> pd.Series | None:
    """The first row of df matching every column == value (None if none)."""
    if df is None or df.empty or any(k not in df for k in kw):
        return None
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m].iloc[0] if m.any() else None


def _ratio(r: pd.Series | None, p: bool = False) -> dict[str, Any]:
    if r is None:
        return {}
    d = dict(ratio=r["ratio"], low=r["ratio_low"], high=r["ratio_high"])
    if p:
        d["p"] = r["p"]
    return d


def _key(text: str) -> str:
    """A stats key (letters, digits, _) from a subset or recording name."""
    text = str(text).replace("-", "m").replace("+", "p").replace(".", "")
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def _rec_key(rec: str) -> str:
    return "_".join(str(rec).split("-")[:3])  # 2020-05-13-10_00 -> 2020_05_13


def _prune(x: Any) -> Any:
    """Drop None, NaN and empty dicts; numpy scalars become Python numbers."""
    if isinstance(x, dict):
        out = {k: _prune(v) for k, v in x.items()}
        return {k: v for k, v in out.items() if v is not None and v != {}}
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
    return x if math.isfinite(x) else None


def _dyad_oe(oe: pd.DataFrame, kind: str, model: str) -> dict[str, Any]:
    t = oe[(oe["kind"] == kind) & (oe["model"] == model)] if len(oe) else oe
    if not len(t):
        return {}
    inf = t[t["informative"].astype(bool)]
    per = {_rec_key(r.recording): dict(observed=r.observed, expected=r.expected, oe=r.oe,
                                       low=r.oe_low, high=r.oe_high,
                                       informative=int(bool(r.informative)))
           for r in t.itertuples()}
    return dict(informative=len(inf), above=int((inf["direction"] == "higher").sum()),
                below=int((inf["direction"] == "lower").sum()), per_dyad=per)


def _kind_stats(kind: str, E: pd.DataFrame, oe: pd.DataFrame, jsum: pd.DataFrame,
                lodo: pd.DataFrame, sens: pd.DataFrame, tests: pd.DataFrame,
                led_dyads: list[str]) -> dict[str, Any]:
    g = _pick(E, kind=kind, backend=PRIMARY, term=TAU_TERM)
    v: dict[str, Any] = {}
    if g is not None:
        v.update(n_events=g["n_events"], n_bins=g["n_bins"], n_dyads=g["n_recordings"])
    v["tau1s"] = dict(glmm=_ratio(g), fe=_ratio(_pick(E, kind=kind, backend=CHECK, term=TAU_TERM),
                                              p=True),
                      re=_ratio(_pick(E, kind=kind, backend=META, term=TAU_TERM), p=True),
                      vb=_ratio(_pick(E, kind=kind, backend=VB, term=TAU_TERM)))
    sd = _pick(E, kind=kind, backend=META, model=TAU_MODEL, term="dyad_sd_sub_effect")
    hg = _pick(E, kind=kind, backend=META, model=TAU_MODEL, term="heterogeneity_G")
    v["heterogeneity"] = dict(dyad_sd=sd["estimate"] if sd is not None else None,
                              dyad_sd_p=sd["p"] if sd is not None else None,
                              g_p=hg["p"] if hg is not None else None)
    v["dyads_tau1s"] = _dyad_oe(oe, kind, TAU_MODEL)
    v["guard"] = dict(glmm=_ratio(_pick(E, kind=kind, backend=PRIMARY, term=GUARD_TERM)),
                      fe=_ratio(_pick(E, kind=kind, backend=CHECK, term=GUARD_TERM), p=True),
                      re=_ratio(_pick(E, kind=kind, backend=META, term=GUARD_TERM), p=True))
    v["dyads_guard"] = _dyad_oe(oe, kind, GUARD_MODEL)
    kj = _pick(E, kind=kind, backend=PRIMARY, term="kernel_sub_joint")
    v["kernel_glmm"] = dict(mean=_ratio(_pick(E, kind=kind, backend=PRIMARY, term="kernel_sub_mean")),
                            mean_late=_ratio(_pick(E, kind=kind, backend=PRIMARY,
                                                   term="kernel_sub_mean_late")),
                            joint_p=kj["p"] if kj is not None else None)
    scan = E[(E["kind"] == kind) & (E["backend"] == PRIMARY)
             & E["term"].astype(str).str.match(r"sub_any_\d+ms$")]
    v["tau_scan_glmm"] = {_key(r.term.removeprefix("sub_any_")): r.ratio for r in scan.itertuples()}
    js = jsum[jsum["kind"] == kind] if len(jsum) else jsum
    v["jitter_p"] = {f"{r.null}_{'tau1s' if str(r.stat).startswith('tau_') else 'kernel'}":
                     r.p_two_sided for r in js.itertuples()}
    if len(js):
        v["jitter_n"] = int(js["n_jitter"].min())
    lk = lodo[lodo["kind"] == kind] if len(lodo) else lodo
    if len(lk):
        v["lodo_fe"] = dict(min=lk["ratio"].min(), max=lk["ratio"].max())
    ss = sens[(sens["kind"] == kind) & (sens["term"] == TAU_TERM)] if len(sens) else sens
    v["sensitivity"] = {}
    for name, grp in (ss.groupby("subset", sort=False) if len(ss) else []):
        fe = _pick(grp, backend=CHECK)
        v["sensitivity"][_key(name)] = dict(
            glmm=_ratio(_pick(grp, backend=PRIMARY)), fe=_ratio(fe, p=True),
            re=_ratio(_pick(grp, backend=META), p=True),
            n_dyads=fe["n_recordings"] if fe is not None else None)
    t = oe[(oe["kind"] == kind) & (oe["model"] == TAU_MODEL)] if len(oe) else oe
    if led_dyads and len(t):
        v["support_led_dyads"] = dict(events=t[t["recording"].isin(led_dyads)]["observed"].sum(),
                                      events_all=t["observed"].sum(), n_dyads=len(led_dyads))
    d = _pick(tests, analysis=f"post_chirp_hazard:{kind}:sub", null="matched") \
        if len(tests) else None
    if d is not None:
        v["descriptive_post_chirp"] = dict(ratio=d["ratio"], low=d["ratio_ci_low"],
                                           high=d["ratio_ci_high"])
    return v


def _sim_stats(sdir: Path, E: pd.DataFrame) -> dict[str, Any]:
    summ = _read(sdir / "sim_summary.csv")
    if summ.empty:
        return {}

    def row(sc: str, kind: str, backend: str = CHECK, term: str = TAU_TERM) -> pd.Series | None:
        return _pick(summ, scenario=sc, kind=kind, term=term, backend=backend)

    def dyad_sd(sc: str, kind: str) -> float | None:
        """True between-dyad SD of the log HR in a scenario (0 = one common effect)."""
        r = row(sc, kind, META, "dyad_sd_sub_effect")
        return r["truth"] if r is not None else None

    v: dict[str, Any] = {}
    for kind in hz_hazard.KINDS:
        k: dict[str, Any] = {}
        # one common effect (recovery) and dyad-specific effects around it (het_recovery)
        for sc in ("recovery", "het_recovery"):
            rec, re_ = row(sc, kind), row(sc, kind, META)
            if rec is not None:
                k[sc] = dict(truth=math.exp(rec["truth"]), fe=rec["ratio_median"],
                             fe_low=rec["ratio_q025"], fe_high=rec["ratio_q975"],
                             power_fe_cluster=rec["reject_cluster"],
                             re=re_["ratio_median"] if re_ is not None else None,
                             power_re=re_["reject_naive"] if re_ is not None else None,
                             dyad_sd=dyad_sd(sc, kind))
        for sc in ("no_effect", "het_no_effect"):
            a, b, c = row(sc, kind), row(sc, kind, PRIMARY), row(sc, kind, META)
            if a is not None:
                k[f"fpr_{sc}"] = dict(fe_wald=a["reject_naive"], fe_cluster=a["reject_cluster"],
                                      glmm=b["reject_naive"] if b is not None else None,
                                      re=c["reject_naive"] if c is not None else None,
                                      dyad_sd=dyad_sd(sc, kind))
        for sc in ("annotation", "detection", "lag_0.25", "lag_0.5", "lag_0.75", "hidden_state",
                   "hidden_state_short"):
            r = row(sc, kind)
            if r is not None:
                k[_key(sc)] = dict(hr=r["ratio_median"], low=r["ratio_q025"], high=r["ratio_q975"],
                                   truth=math.exp(r["truth"]))
        v[kind] = k
    f = _pick(E, kind="offset", backend=CHECK, term=TAU_TERM)
    if f is not None:
        phi, frac = hz_sim.trigger_match(summ, _read(sdir / "sim_trigger_meta.csv"),
                                         math.log(float(f["ratio"])))
        v["offset"]["trigger_share"] = phi
        v["offset"]["trigger_offsets_frac"] = frac
    return v


def _acausal_stats(adir: Path, sdir: Path) -> dict[str, Any]:
    A = _read(adir / "acausal_effects.csv")
    if A.empty:
        return {}
    v: dict[str, Any] = {}
    for kind in hz_hazard.KINDS:
        T = A[(A["variant"] == "two_sided") & (A["kind"] == kind)]
        k: dict[str, Any] = {}
        for lab, model, pt, ft in (("pf_1s", "pf_1000ms", "sub_any_1000ms", "sub_fut_1000ms"),
                                   ("pf_guard", "pf_guard_500ms", GUARD_TERM, "sub_fgd_500ms")):
            k[lab] = dict(ratio=_ratio(_pick(T, model=model, backend=CHECK, term=f"{pt}-{ft}"),
                                       p=True),
                          hr_past=_ratio(_pick(T, model=model, backend=CHECK, term=pt)),
                          hr_future=_ratio(_pick(T, model=model, backend=CHECK, term=ft)),
                          glmm=_ratio(_pick(T, model=model, backend=PRIMARY, term=f"{pt}-{ft}")))
        k["kernel"] = dict(
            ratio=_ratio(_pick(T, model="kernel_two_sided", backend=CHECK,
                               term="kernel_sub_mean-kernel_sub_fut_mean"), p=True),
            before=_ratio(_pick(T, model="kernel_two_sided", backend=CHECK, term="kernel_sub_mean")),
            after=_ratio(_pick(T, model="kernel_two_sided", backend=CHECK,
                               term="kernel_sub_fut_mean")))
        if kind == "offset":
            B = A[A["kind"] == "offset"]
            for w in ("w_2000ms", "w_5000ms"):
                k[f"mirror_{w}"] = {
                    lab: _ratio(_pick(B, variant="restricted_real-mirror", subset=w, term=term),
                                p=True)
                    for lab, term in (("tau1s", TAU_TERM), ("guard", GUARD_TERM))}
        v[kind] = k
    for name, stats, kind in (("acausal_sim_summary.csv", ACSIM_OFFSET_STATS, "offset"),
                              ("acausal_sim_onset_summary.csv", ACSIM_ONSET_STATS, "onset")):
        S = _read(sdir / name)
        if S.empty:
            continue
        sim: dict[str, Any] = {}
        for r in S[S["stat"].isin(stats)].itertuples():
            sim.setdefault(r.scenario, {})[r.stat] = dict(
                median=r.median, q025=r.q025, q975=r.q975, observed=r.observed,
                exceed_frac=r.exceed_frac, p_obs=r.p_obs)
        v.setdefault(kind, {})["simulation"] = sim
    return v


def chase_hazard_stats(out: Path, trials: list[Trial], flags: pd.DataFrame, cfg: Config,
                       led_dyads: list[str]) -> dict[str, Any]:
    """Every citable number of one run, from the tables in ``out``."""
    out = Path(out)
    h = out / "hazard"
    E = _read(h / "hazard_effects.csv")
    if E.empty:
        raise FileNotFoundError(f"no {h / 'hazard_effects.csv'}")
    oe, jsum = _read(h / "hazard_dyad_oe.csv"), _read(h / "hazard_jitter_summary.csv")
    lodo, sens = _read(h / "hazard_leave_one_dyad_out.csv"), _read(h / "hazard_sensitivity.csv")
    tests = _read(out / "descriptives" / "tests.csv")
    fish = {(tr.group, tr.dom_fish) for tr in trials} | {(tr.group, tr.sub_fish) for tr in trials}
    data = dict(
        n_dyads=len(trials), n_fish=len(fish),
        n_chases=sum(int((tr.chases[:, 0] < tr.obs_end).sum()) for tr in trials),
        n_sub_chirps=sum(int(tr.valid(tr.chirps_sub).sum()) for tr in trials),
        n_sub_chirps_in_chase=sum(int((tr.valid(tr.chirps_sub) & tr.in_chase(tr.chirps_sub)).sum())
                                  for tr in trials),
        n_dom_chirps=sum(int(tr.valid(tr.chirps_dom).sum()) for tr in trials),
        window_h=(cfg.max_time or float("nan")) / 3600,
        bin_width_s=cfg.bin_width,
    )
    un = flags[(flags["category"] == "chirp_unassigned_upstream")
               & flags["recording"].isin({tr.recording for tr in trials})]
    data["n_unassigned_distinct_window"] = un["value"].sum() if len(un) else 0
    values = dict(data=data)
    for kind in hz_hazard.KINDS:
        values[kind] = _kind_stats(kind, E, oe, jsum, lodo, sens, tests, led_dyads)
    values["simulation"] = _sim_stats(out / "simulation", E)
    values["acausal"] = _acausal_stats(out / "acausal", out / "simulation")
    return _prune(values)
