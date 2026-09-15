"""Example 103 -- "Crease Press": a cyclic hydraulic press creasing a mixed stack.

Round-7 perf benchmark (`crease-press`). Seven clamped sheets are stacked with
layer gaps and creased twice by one kinematic press bar at two lateral offsets,
so the creases form in two places and the internal-friction hinges run
hysteresis loops (crease, partially unload, re-crease elsewhere):

  3 denim sheets    StrainLimitingBaraffWitkinShell membrane
                    + DahlFrictionDiscreteShellBending (internal friction)
                    -- the cloth-dataset towel values (physics/fixtures/
                    tumbler_stay.py): kappa=2e-5, ell=0.3, m_hat=8e-3,
                    thickness 0.8 mm, areal density 0.30 kg/m^2. Finest mesh:
                    these carry the round's main target kernel.
  2 cardboard       NeoHookeanShell membrane
                    + StrainPlasticDiscreteShellBending
  2 sheet-metal     NeoHookeanShell membrane
                    + StressPlasticDiscreteShellBending
                    -- 101_press's values: E=8e4, thickness 2.5 mm,
                    bending stiffness 4e3, yield strain 0.02 rad,
                    yield stress 250.

The denim is interleaved with the cardboard/metal through the stack so every
sheet sees neighbours of the other family. Every sheet is clamped along the two
edges parallel to the press (blank holder, builtin.is_fixed), exactly as
101_press does. The observable is the residual crease after retraction: the
denim keeps its creases (Dahl friction holds them), the cardboard and metal
keep plastic creases; the elastic part of the schedule is identical for all.

Headless:  python main.py [FRAMES] [--verify] [--result JSON] [--capture DIR]
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from asset_dir import AssetDir
import verify as crease_verify   # bind BEFORE r6_common: it inserts examples/ AND
                                 # 95_tumbler_garments/ at sys.path[0:2], and the
                                 # tumbler has its own verify.py
import r6_common as r6

from uipc import Animation, Logger, Matrix4x4, view
import uipc.builtin as builtin
from uipc.core import Engine, Scene, World
from uipc.geometry import (SimplicialComplexIO, ground, trimesh, label_surface,
                           label_triangle_orient, flip_inward_triangles)
from uipc.constitution import (AffineBodyConstitution, DahlFrictionDiscreteShellBending,
                               ElasticModuli2D, NeoHookeanShell, SoftTransformConstraint,
                               StrainLimitingBaraffWitkinShell,
                               StrainPlasticDiscreteShellBending,
                               StressPlasticDiscreteShellBending)

SCENE_NAME = "103_crease_press"
TETMESH = AssetDir.tetmesh_path()

# ---------------------------------------------------------------------------
# schedule / sizes.  dt starts from 101_press (0.01) and was calibrated to
# 1/60 (see the round-7 s00 size sweep): at 0.01 the soft denim needs >10
# Newton iterations per frame through the press phases and the run costs
# 3x the wall budget; 1/60 keeps both press cycles inside the Newton budget
# with margin (all frames converged, no line-search limit hits).  The default
# tol_rate is likewise calibrated from 101_press's 1e-5 down to 1e-4: the
# tighter value doubles the PCG bill for observables that do not move (the
# sweep records the comparison).
# ---------------------------------------------------------------------------
DT = float(os.environ.get("CP_DT", 1.0 / 60.0))
DEFAULT_FRAMES = 130

# sheet stack (all sheets the same rectangular footprint, interleaved families)
SHEET_X = 0.48                   # extent along the press bar (m), free edges
SHEET_Z = 0.80                   # extent across the press (m), clamped at +-SHEET_Z/2
STACK_Y0 = 0.50                  # rest height of the bottom sheet's mid-surface

# denim (cloth-dataset tumbler_stay.py towel values, verbatim)
DENIM_THICKNESS = 0.0008         # one-sided shell thickness (m)
DENIM_AREAL = 0.30               # kg/m^2
DENIM_KAPPA = 2.0e-5             # bending stiffness (N*m)
DENIM_ELL = 0.3                  # Dahl transition angle (rad)
DENIM_M_HAT = 8.0e-3             # saturated friction moment per length (N*m/m)
DENIM_STRETCH_E = 4000.0 * (1.0 - 0.09) / (2.0 * DENIM_THICKNESS)
DENIM_SHEAR_E = 40.0 * 2.0 * 1.3
DENIM_POISSON = 0.3
DENIM_STRAIN_RATE = 100.0

# cardboard / sheet metal (101_press values, verbatim, except the yield
# thresholds -- see below)
CARTON_YOUNG = 8.0e4
CARTON_POISSON = 0.35
CARTON_DENSITY = 200.0
CARTON_THICKNESS = 2.5e-3        # one-sided (m)
CARTON_BEND = 4.0e3
# The yield thresholds are 101_press's, carried over as the same yield
# CURVATURE: 101_press's 0.02 rad sits on 50 mm elements (0.4 1/m), so on
# this scene's 15.5 mm elements the same material yields at
# 0.02 * (15.5/50) ~ 0.006 rad per hinge (the yield threshold is
# set for this element size; if --carton-edge moves, move it with the same
# rule).  The stress threshold stays at
# 101_press's 250 verbatim -- at this mesh it resolves to the same per-hinge
# angle (theta_y = yield_stress * h_bar / (2 * kappa * L0) ~ 0.005 rad), so
# both plastic laws yield together, as in 101_press.
CARTON_YIELD_STRAIN = float(os.environ.get("CP_YSTRAIN", 0.006))  # rad per hinge
CARTON_YIELD_STRESS = float(os.environ.get("CP_YSTRESS", 250.0))
CARTON_HARDENING = 0.0

# press bar (an ABD cube driven by SoftTransformConstraint, as 101_press).
# Two calibration lessons are baked into these numbers (see the size sweep):
# 1. The bar's face must stop just below the stack's rest surface.  The crease
#    hinges yield (that is the point of the plastic sheets), so once the fold
#    forms the stack's moment resistance collapses; a face that keeps driving
#    80 mm past that point buckles the whole stack through and the sheets
#    plunge under the bar.
# 2. Whatever depth the fold does reach must stay small against the clamp
#    span: the in-plane distance clamp->crease grows like
#    sqrt(span^2 + depth^2), and that membrane tension sheared the
#    clamp-corner triangles of the stiff carton sheets to 30x rest area in
#    early drafts (sheets puckered near the clamps and slid 10 cm sideways).
# The ground halfplane doubles as the die: it sits 28 mm under the stack's
# rest plane, so the crease strip bottoms out on the bed and the kink
# LOCALISES over the bar's footprint (a wide dome cannot: the bed is flat and
# close).  The face stops ~2 mm above the bed + the compressed stack, so the
# end-stroke squeezes the layer gaps out without crushing the sheets.
PRESS_HALF_X = 0.45              # spans every sheet in x with margin
PRESS_HALF_H = 0.075
PRESS_HALF_Z = 0.04              # crease-line width (a sharp V, not a wide strip)
PRESS_TOP_FACE = 0.62            # bar bottom face at rest, clear of the stack
PRESS_DEPTH = float(os.environ.get("CP_PDEPTH", 0.10))    # face travel below the stack's rest top surface
DIE_Y = float(os.environ.get("CP_DIE", STACK_Y0 - 0.075))  # ground halfplane = the press die (75 mm under the rest plane)
Z_OFF = 0.08                     # lateral offsets of the two crease lines

# schedule (fractions of the run). Two full press/hold/lift cycles with a
# lateral shift between them, then settle: the residual creases ARE the
# physics being exercised.
F = (0.13, 0.05, 0.11, 0.06, 0.13, 0.05, 0.11)     # p1,h1,l1,shift,p2,h2,l2

# contact
R_NOMINAL_DENIM = DENIM_THICKNESS
R_NOMINAL_CARTON = CARTON_THICKNESS
D_HAT_NOMINAL = 2.0e-3
CONTACT_RESISTANCE = 1.0e9
MU_TOOL_SHEET = 0.15             # 101_press's tool/sheet pair
MU_SHEET_SHEET = 0.20            # 101_press's sheet/sheet pair
LAYER_MARGIN = 2.0e-3

ap = r6.build_argparser(SCENE_NAME, DEFAULT_FRAMES)
r6.add_r6_args(ap, 0.011)
ap.add_argument("--carton-edge", type=float, default=0.0155,
                help="target cardboard/metal element size (m)")
ap.add_argument("--verify", action="store_true",
                help="run the physical-soundness + constitution-state audit")
args = ap.parse_args()
r6.configure_benchmark_timers()
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Error")))

N = args.frames
_phase_len = [max(1, int(f * N)) for f in F]


def make_sheet_grid(edge: float):
    """Rectangular vertex grid (SHEET_Z rows of SHEET_X) + 2-triangle-per-cell
    faces at the target element size.  Returns (res_x, res_z, V, F)."""
    rx = max(3, int(round(SHEET_X / edge)) + 1)
    rz = max(3, int(round(SHEET_Z / edge)) + 1)
    xs = np.linspace(-0.5 * SHEET_X, 0.5 * SHEET_X, rx)
    zs = np.linspace(-0.5 * SHEET_Z, 0.5 * SHEET_Z, rz)
    V = np.array([[x, 0.0, z] for z in zs for x in xs], dtype=np.float64)
    F = []
    for j in range(rz - 1):
        for i in range(rx - 1):
            a = j * rx + i
            F += [[a, a + 1, a + rx + 1], [a, a + rx + 1, a + rx]]
    return rx, rz, V, np.asarray(F, dtype=np.int64)


def hinge_stencils(rx: int, rz: int):
    """Interior-edge bending hinges [x0,x1,x2,x3] of the grid (x1,x2 = the
    edge), matching the engine's own hinge census for a flat grid sheet."""
    idx = lambda i, j: j * rx + i
    st = []
    for j in range(1, rz - 1):                       # edges along x
        for i in range(rx - 1):
            st.append([idx(i, j - 1), idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
    for i in range(1, rx - 1):                       # edges along z
        for j in range(rz - 1):
            st.append([idx(i - 1, j), idx(i, j), idx(i, j + 1), idx(i + 1, j + 1)])
    for j in range(rz - 1):                          # diagonals (all interior)
        for i in range(rx - 1):
            st.append([idx(i + 1, j), idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return np.asarray(st, dtype=np.int64)


# --- build the stack ---------------------------------------------------------
# interleaved bottom-up: denim, cardboard, denim, metal, denim, cardboard, metal
STACK = [("denim", "dahl"), ("cardboard", "strain"), ("denim", "dahl"),
         ("metal", "stress"), ("denim", "dahl"), ("cardboard", "strain"),
         ("metal", "stress")]

denim_rx, denim_rz, denim_V, denim_F = make_sheet_grid(args.edge_len)
carton_rx, carton_rz, carton_V, carton_F = make_sheet_grid(args.carton_edge)


def tri_min_height(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    e = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                  np.linalg.norm(a - c, axis=1)], axis=1)
    return float((2.0 * area[:, None] / np.maximum(e, 1e-30)).min())


MIN_TRI_HEIGHT = min(tri_min_height(denim_V, denim_F), tri_min_height(carton_V, carton_F))
# dataset rule (r6_common.contact_resolution): r <= 0.25 h_min, d_hat <= 0.3 h_min,
# 2r + d_hat <= 0.8 h_min.  A mixed-thickness stack has to read the rule PER
# FAMILY: the floor exists to keep one sheet's own triangles out of permanent
# self-contact, so each family's 2r+d_hat is checked against its own h_min
# (denim 0.8 mm shells vs the fine mesh, carton 2.5 mm shells vs the coarse
# one), while d_hat -- a single engine-wide number -- is capped against the
# global minimum so the finest sheet stays conservative.
D_HAT = min(D_HAT_NOMINAL, 0.3 * MIN_TRI_HEIGHT)
for name, r, h in (("denim", R_NOMINAL_DENIM, tri_min_height(denim_V, denim_F)),
                   ("cardboard/metal", R_NOMINAL_CARTON, tri_min_height(carton_V, carton_F))):
    if r > 0.25 * h or 2 * r + D_HAT > 0.8 * h:
        raise SystemExit(f"contact-resolution rule violated for {name} "
                         f"(family h_min={h*1e3:.2f} mm, global h_min={MIN_TRI_HEIGHT*1e3:.2f} mm); "
                         f"coarsen --edge-len / --carton-edge")

GAP = R_NOMINAL_DENIM + R_NOMINAL_CARTON + D_HAT + LAYER_MARGIN   # worst adjacent pair

# --- scene / engine ----------------------------------------------------------
workspace = args.workspace or AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = DT
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = D_HAT
config["line_search"]["max_iter"] = 12
# thin shells next to a stiff kinematic affine body: the relative PCG stopping
# criterion has to be tightened or the shells' residuals drown under the stiff
# body's scale (101_press's note; calibrated 1e-5 -> 1e-4 in the round-7 s00
# sweep -- same observables, half the PCG bill)
config["linear_system"]["tol_rate"] = args.tol_rate if args.tol_rate is not None else 1.0e-4
config["linear_system"]["fem_preconditioner"] = "mas"
scene = Scene(config)

ct = scene.contact_tabular()
ct.default_model(MU_SHEET_SHEET, CONTACT_RESISTANCE)
elem_tool = ct.create("tool")
elem_sheet = ct.create("sheet")
ct.insert(elem_tool, elem_sheet, MU_TOOL_SHEET, CONTACT_RESISTANCE)
ct.insert(elem_sheet, elem_sheet, MU_SHEET_SHEET, CONTACT_RESISTANCE)

slbws = StrainLimitingBaraffWitkinShell()
denim_bending = DahlFrictionDiscreteShellBending()
neo_shell = NeoHookeanShell()
strain_plastic = StrainPlasticDiscreteShellBending()
stress_plastic = StressPlasticDiscreteShellBending()
abd = AffineBodyConstitution()
stc = SoftTransformConstraint()

SHEETS = []
for k, (name, kind) in enumerate(STACK):
    denim = kind == "dahl"
    if denim:
        rx, rz, V0, F = denim_rx, denim_rz, denim_V, denim_F
    else:
        rx, rz, V0, F = carton_rx, carton_rz, carton_V, carton_F
    V = V0.copy()
    V[:, 1] = STACK_Y0 + k * GAP
    mesh = trimesh(np.ascontiguousarray(V), np.ascontiguousarray(F, dtype=np.int32))
    label_surface(mesh)
    if denim:
        slbws.apply_to(mesh,
                       stretch_moduli=ElasticModuli2D.youngs_poisson(DENIM_STRETCH_E, DENIM_POISSON),
                       shear_moduli=ElasticModuli2D.youngs_poisson(DENIM_SHEAR_E, DENIM_POISSON),
                       mass_density=DENIM_AREAL / (2.0 * DENIM_THICKNESS),
                       thickness=DENIM_THICKNESS,
                       strain_rate=DENIM_STRAIN_RATE)
        denim_bending.apply_to(mesh, DENIM_KAPPA, DENIM_M_HAT, DENIM_ELL)
    else:
        neo_shell.apply_to(mesh, ElasticModuli2D.youngs_poisson(CARTON_YOUNG, CARTON_POISSON),
                           CARTON_DENSITY, CARTON_THICKNESS)
        if kind == "strain":
            strain_plastic.apply_to(mesh, CARTON_BEND, CARTON_YIELD_STRAIN, CARTON_HARDENING)
        else:
            stress_plastic.apply_to(mesh, CARTON_BEND, CARTON_YIELD_STRESS, CARTON_HARDENING)
    elem_sheet.apply_to(mesh)
    # blank holder: clamp the two edges parallel to the press (z = +-SHEET_Z/2)
    fixed = mesh.vertices().find(builtin.is_fixed)
    if fixed is None:
        fixed = mesh.vertices().create(builtin.is_fixed, 0)
    fv = view(fixed)
    edge = np.abs(V[:, 2]) > 0.5 * SHEET_Z - 1e-9
    fv[:] = edge.astype(np.int32)
    obj = scene.objects().create(f"sheet_{k}_{name}")
    slot = obj.geometries().create(mesh)[0]
    SHEETS.append(dict(k=k, name=name, kind=kind, slot=slot,
                       gid=obj.geometries().ids()[0], V0=V, F=F, rest_y=V[0, 1],
                       stencils=hinge_stencils(rx, rz)))

N_HINGES = {"dahl": 0, "strain": 0, "stress": 0}
for s in SHEETS:
    N_HINGES[s["kind"]] += len(s["stencils"])
N_TRIS = sum(len(s["F"]) for s in SHEETS)
N_VERTS = sum(len(s["V0"]) for s in SHEETS)

# --- press bar ---------------------------------------------------------------
press_pre = r6.transform4(scale=(2 * PRESS_HALF_X, 2 * PRESS_HALF_H, 2 * PRESS_HALF_Z))
press_mesh = SimplicialComplexIO(press_pre).read(f"{TETMESH}/cube.msh")
label_surface(press_mesh)
label_triangle_orient(press_mesh)
press_mesh = flip_inward_triangles(press_mesh)
abd.apply_to(press_mesh, 2.0e7)
stc.apply_to(press_mesh, np.array([3600.0, 120.0], dtype=np.float64))
elem_tool.apply_to(press_mesh)
press_mesh.instances().resize(1)
r6.set_instance_transform(press_mesh, 0, r6.transform4(translate=(0.0, PRESS_TOP_FACE + PRESS_HALF_H, 0.0)))
press_obj = scene.objects().create("press")
press_slot = press_obj.geometries().create(press_mesh)[0]

STACK_TOP = STACK_Y0 + 6 * GAP + CARTON_THICKNESS   # top sheet's rest surface
Y_TOP_CENTER = STACK_TOP + 0.03 + PRESS_HALF_H      # clear of the stack at rest
Y_BOTTOM_CENTER = STACK_TOP - PRESS_DEPTH + PRESS_HALF_H


def press_pose(frame: int):
    """(y_center, z_center) of the bar at the END of `frame` -- press/hold/lift
    cycle 1 at z=-Z_OFF, lateral shift at the top, cycle 2 at z=+Z_OFF, settle."""
    def smooth(a, b, t):
        t = min(max(t, 0.0), 1.0)
        return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * t))
    p1, h1, l1, sh, p2, h2, l2 = _phase_len
    y_hi, y_lo = Y_TOP_CENTER, Y_BOTTOM_CENTER
    f = frame
    if f < p1:
        return smooth(y_hi, y_lo, f / max(p1 - 1, 1)), -Z_OFF
    f -= p1
    if f < h1:
        return y_lo, -Z_OFF
    f -= h1
    if f < l1:
        return smooth(y_lo, y_hi, f / max(l1 - 1, 1)), -Z_OFF
    f -= l1
    if f < sh:
        return y_hi, smooth(-Z_OFF, Z_OFF, f / max(sh - 1, 1))
    f -= sh
    if f < p2:
        return smooth(y_hi, y_lo, f / max(p2 - 1, 1)), Z_OFF
    f -= p2
    if f < h2:
        return y_lo, Z_OFF
    f -= h2
    if f < l2:
        return smooth(y_lo, y_hi, f / max(l2 - 1, 1)), Z_OFF
    return y_hi, Z_OFF


def phase(frame: int) -> str:
    p1, h1, l1, sh, p2, h2, l2 = _phase_len
    f = frame
    for name, ln in (("press1", p1), ("hold1", h1), ("lift1", l1), ("shift", sh),
                     ("press2", p2), ("hold2", h2), ("lift2", l2)):
        if f < ln:
            return name
        f -= ln
    return "settle"


def animate_press(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    view(geo.instances().find(builtin.is_constrained))[0] = 1
    y, z = press_pose(info.frame())
    T = Matrix4x4.Identity()
    T[1, 3] = y
    T[2, 3] = z
    view(geo.instances().find(builtin.aim_transform))[0] = T


scene.animator().insert(press_obj, animate_press)

ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(DIE_Y))                   # the die bed

world.init(scene)
init_sanity = r6.run_sanity_check(world) if not args.no_sanity else {"skipped": True}
print("SANITY(init):", init_sanity.get("result"), "clean=", init_sanity.get("clean"), flush=True)
if not world.is_valid():
    raise SystemExit("world invalid after init -- sanity check failed")

print(f"crease-press: {len(SHEETS)} sheets, {N_VERTS} verts, {N_TRIS} tris, "
      f"hinges dahl/strain/stress = {N_HINGES['dahl']}/{N_HINGES['strain']}/{N_HINGES['stress']}, "
      f"denim e={args.edge_len*1e3:.1f}mm res={denim_rx}x{denim_rz}, "
      f"carton e={args.carton_edge*1e3:.1f}mm res={carton_rx}x{carton_rz}, "
      f"h_min={MIN_TRI_HEIGHT*1e3:.2f}mm, d_hat={D_HAT*1e3:.2f}mm, "
      f"gap={GAP*1e3:.1f}mm, dt={DT:g}, {N} frames", flush=True)

capture = r6.Capture(args.capture, scene, up_dir="y_up",
                     ground_height=DIE_Y,
                     stride=args.capture_stride, obj_dir=args.obj)
capture.bind([(f"sheet{s['k']}_{s['name']}", s["slot"]) for s in SHEETS] + [("press", press_slot)])

runner = r6.Runner(world, engine, scene, args, SCENE_NAME)

audit = None
if args.verify:
    Audit = crease_verify.Audit
    audit = Audit(SHEETS, DT, D_HAT,
                  {"dahl": (DENIM_KAPPA, DENIM_M_HAT, DENIM_ELL),
                   "strain": (CARTON_BEND, CARTON_YIELD_STRAIN),
                   "stress": (CARTON_BEND, CARTON_YIELD_STRESS)},
                  press_pose, STACK_Y0, DIE_Y, Z_OFF, SHEET_X, SHEET_Z)  # DIE_Y = ground
    audit.observe(0, [np.asarray(s["slot"].geometry().positions().view()).reshape(-1, 3)
                      for s in SHEETS], press_pose(0), None)


def measure():
    return {"frame": world.frame(), "phase": phase(world.frame() - 1),
            "press_y": float(press_pose(world.frame())[0]),
            "press_z": float(press_pose(world.frame())[1])}


capture.write_frame(0, force=True)
for i in range(N):
    ms = runner.step(on_frame=measure)
    if audit is not None:
        audit.observe(world.frame(),
                      [np.asarray(s["slot"].geometry().positions().view()).reshape(-1, 3)
                       for s in SHEETS],
                      press_pose(world.frame()), runner.frame_stats[-1])
    capture.write_frame(i + 1)
    if i % 20 == 0 or i == N - 1:
        st = runner.frame_stats[-1]
        t = runner.trace[-1] if runner.trace else measure()
        extra = ""
        if audit is not None:
            extra = f" dahlF={audit.state_summary()['dahl_active_frac']:.3f} " \
                    f"yield={audit.state_summary()['strain_yield_frac']:.3f}/" \
                    f"{audit.state_summary()['stress_yield_frac']:.3f}"
        print(f"frame {i:4d} {ms:7.1f}ms {t['phase']:7s} newton={st['newton_iterations']:2d} "
              f"pcg={st['linear_solver_iterations']:5d} ls={st['line_search_trials']:2d} "
              f"conv={st['converged']} press_y={t['press_y']:.3f} press_z={t['press_z']:+.3f}"
              f"{extra}", flush=True)

final_sanity = r6.run_sanity_check(world) if not args.no_sanity else {"skipped": True}
capture.finish(runner.frame_ms)

# --- observables -------------------------------------------------------------
last_press_y, last_press_z = press_pose(N)
obs = {"frames": N, "dt": DT, "n_tris": N_TRIS, "n_verts": N_VERTS,
       "n_hinges_dahl": N_HINGES["dahl"], "n_hinges_strain": N_HINGES["strain"],
       "n_hinges_stress": N_HINGES["stress"],
       "denim_edge": args.edge_len, "carton_edge": args.carton_edge,
       "d_hat": D_HAT, "gap": GAP,
       "press_final_y": last_press_y, "press_final_z": last_press_z}
if audit is None:
    obs["all_finite"] = True
if audit is not None:
    obs.update(audit.summary())

r6.emit(runner, obs, {"sanity": {"init": init_sanity, "final": final_sanity},
                      "audit_rows": audit.rows if audit is not None else None,
                      "config": {"dt": DT, "sheet_x": SHEET_X, "sheet_z": SHEET_Z, "stack_y0": STACK_Y0,
                                 "press_depth": PRESS_DEPTH, "z_off": Z_OFF, "press_half_z": PRESS_HALF_Z,
                                 "denim": {"kappa": DENIM_KAPPA, "m_hat": DENIM_M_HAT,
                                           "ell": DENIM_ELL, "thickness": DENIM_THICKNESS},
                                 "carton": {"young": CARTON_YOUNG, "bend": CARTON_BEND,
                                            "yield_strain": CARTON_YIELD_STRAIN,
                                            "yield_stress": CARTON_YIELD_STRESS}}})
if audit is not None:
    audit.report()
