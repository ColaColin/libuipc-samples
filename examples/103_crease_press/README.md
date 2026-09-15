# 103 — Crease Press (round-7 `crease-press` benchmark)

A cyclic hydraulic press creasing a mixed stack of seven clamped sheets, built
as the round-7 performance workload for the three shell-bending constitutions
no previous benchmark executed:

| sheets | membrane | bending | values |
|---|---|---|---|
| 3 denim (finest mesh) | `StrainLimitingBaraffWitkinShell` | `DahlFrictionDiscreteShellBending` | cloth-dataset towel (`physics/fixtures/tumbler_stay.py`): kappa 2e-5, ell 0.3, m_hat 8e-3, thickness 0.8 mm, areal 0.30 kg/m² |
| 2 cardboard | `NeoHookeanShell` | `StrainPlasticDiscreteShellBending` | 101_press: E 8e4, thickness 2.5 mm, bend 4e3, yield carried over as the same yield *curvature* (see `main.py`) |
| 2 sheet metal | `NeoHookeanShell` | `StressPlasticDiscreteShellBending` | 101_press: yield stress 250 |

The denim is interleaved with the cardboard/metal through the stack. Every
sheet is clamped along the two z edges (blank holder, `builtin.is_fixed`, as
`101_press`). One kinematic ABD press bar (cube + `SoftTransformConstraint`,
smooth press/hold/lift as `101_press`) runs **two full cycles at two lateral
offsets** (z = ±80 mm), so creases form in two places and the dahl hinges run
hysteresis loops: crease, partially unload, re-crease elsewhere. The ground
halfplane is the press die that arrests the fold.

The observables: after retraction every sheet keeps a residual crease
(~35-45 mm) — the denim through dahl internal friction, the cardboard/metal
through plastic rest angles.

```
python main.py                 # 130 frames, benchmark output
python main.py --verify        # + physical-soundness and constitution-state audit
python main.py 260 CP-free     # sweeps: --edge-len, --carton-edge, --tol-rate,
                               # CP_DT / CP_PDEPTH / CP_DIE / CP_YSTRAIN / CP_YSTRESS
```

`verify.py` audits finiteness, inversion/collapse, containment, Newton health,
and — the point of the scene — replicates the engine's per-frame constitution
commit kernels in numpy (dahl `F_commit`, plastic `theta_bar`) hinge for hinge,
so "the friction state actually evolves" and "the plastic sheets actually
yield" are measured, not assumed.

Calibration story (element sizes, dt, press depth, die depth, yield
thresholds): `agent_docs/performance/data/2026-09-15-round7-s00/size_sweep.md`
in the main repository.
