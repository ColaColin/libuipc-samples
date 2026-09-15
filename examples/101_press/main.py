"""Example 101 -- "Three-Sheet Press" (elastic vs strain-plastic vs stress-plastic).

A single rigid affine-body punch bar is driven kinematically (SoftTransformConstraint
+ animator) down across *three* identical clamped metal sheets and then retracted.
The only difference between the sheets is the bending model:

  sheet A  DiscreteShellBending                 -- purely elastic bending
  sheet B  StrainPlasticDiscreteShellBending    -- plastic above a curvature/strain threshold
  sheet C  StressPlasticDiscreteShellBending    -- plastic above a bending-moment (stress) threshold

Membrane response for all three is the same NeoHookeanShell; all three are clamped
along the two edges parallel to the punch (builtin.is_fixed, a "blank holder"), and
all three see the *same* punch at the *same* time, so the comparison is controlled.

Physical idea: a stamping press. Everything bends the same on the way down; what
differs is what is left behind when the tool comes back up. The observable is the
residual centre-line deflection after retraction and settling: ~0 for the elastic
sheet, a permanent crease for the two plastic ones.

Headless:  python main.py [FRAMES] [--capture DIR] [--result JSON]
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from asset_dir import AssetDir
import r6_common as r6

from uipc import Animation, Logger, Matrix4x4, Transform, Vector3, view
import uipc.builtin as builtin
from uipc.core import Engine, Scene, World
from uipc.geometry import (SimplicialComplexIO, ground, trimesh, label_surface,
                           label_triangle_orient, flip_inward_triangles)
from uipc.constitution import (AffineBodyConstitution, DiscreteShellBending,
                               ElasticModuli2D, NeoHookeanShell, SoftTransformConstraint,
                               StrainPlasticDiscreteShellBending,
                               StressPlasticDiscreteShellBending)

SCENE_NAME = "101_press"
TETMESH = AssetDir.tetmesh_path()
DT = 0.01

# sheet geometry / material (values follow the repo's own plastic crease demos)
RES = 25                 # 25 x 25 vertices per sheet
SHEET = 1.20             # side length
SHEET_Y = 0.50
SHELL_YOUNG = 8.0e4
SHELL_POISSON = 0.35
SHELL_DENSITY = 200.0
SHELL_THICKNESS = 2.5e-3
BENDING_STIFFNESS = 4.0e3
YIELD_STRESS = float(os.environ.get("YSTRESS", 250.0))          # StressPlastic: bending moment threshold
YIELD_STRAIN = float(os.environ.get("YSTRAIN", 0.02))          # StrainPlastic: per-edge dihedral-angle threshold [rad]
HARDENING = 0.0

SHEET_X = (-1.40, 0.0, 1.40)
PUNCH_HALF_X = 2.25
PUNCH_Y_TOP = 0.62
PUNCH_Y_BOTTOM = 0.235
PUNCH_HALF_H = 0.09

# schedule (fractions of the run)
F_PRESS, F_HOLD, F_LIFT = 0.32, 0.12, 0.26      # rest = settle

ap = r6.build_argparser(SCENE_NAME, 150)
args = ap.parse_args()
r6.configure_benchmark_timers()
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Error")))

N = args.frames
N_PRESS = max(1, int(F_PRESS * N))
N_HOLD = max(1, int(F_HOLD * N))
N_LIFT = max(1, int(F_LIFT * N))

workspace = args.workspace or AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = DT
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.005
config["line_search"]["max_iter"] = 12
# thin shells next to a stiff kinematic affine body: the relative PCG stopping
# criterion has to be tightened or the shells' residuals drown under the stiff
# body's scale (see the note in example 100's config)
config["linear_system"]["tol_rate"] = float(os.environ.get("TOLRATE", 1e-5))
scene = Scene(config)

ct = scene.contact_tabular()
ct.default_model(0.20, 1.0e9)
elem_tool = ct.create("tool")
elem_sheet = ct.create("sheet")
ct.insert(elem_tool, elem_sheet, 0.15, 1.0e9)
ct.insert(elem_sheet, elem_sheet, 0.20, 1.0e9)

shell = NeoHookeanShell()
elastic_bending = DiscreteShellBending()
strain_plastic = StrainPlasticDiscreteShellBending()
stress_plastic = StressPlasticDiscreteShellBending()
abd = AffineBodyConstitution()
stc = SoftTransformConstraint()


def make_sheet(res: int, size: float, cx: float, y: float):
    xs = np.linspace(-0.5 * size, 0.5 * size, res) + cx
    zs = np.linspace(-0.5 * size, 0.5 * size, res)
    V = np.array([[x, y, z] for z in zs for x in xs], dtype=np.float64)
    F = []
    for j in range(res - 1):
        for i in range(res - 1):
            a = j * res + i
            F += [[a, a + 1, a + res + 1], [a, a + res + 1, a + res]]
    m = trimesh(V, np.asarray(F, dtype=np.int32))
    label_surface(m)
    return m, V


SHEETS = []
for idx, (cx, kind) in enumerate(zip(SHEET_X, ("elastic", "strain_plastic", "stress_plastic"))):
    mesh, V0 = make_sheet(RES, SHEET, cx, SHEET_Y)
    shell.apply_to(mesh, ElasticModuli2D.youngs_poisson(SHELL_YOUNG, SHELL_POISSON),
                   SHELL_DENSITY, SHELL_THICKNESS)
    if kind == "elastic":
        elastic_bending.apply_to(mesh, BENDING_STIFFNESS)
    elif kind == "strain_plastic":
        strain_plastic.apply_to(mesh, BENDING_STIFFNESS, YIELD_STRAIN, HARDENING)
    else:
        stress_plastic.apply_to(mesh, BENDING_STIFFNESS, YIELD_STRESS, HARDENING)
    elem_sheet.apply_to(mesh)
    # blank holder: clamp the two edges parallel to the punch (z = +-size/2)
    fixed = mesh.vertices().find(builtin.is_fixed)
    if fixed is None:
        fixed = mesh.vertices().create(builtin.is_fixed, 0)
    fv = view(fixed)
    edge = np.abs(V0[:, 2]) > 0.5 * SHEET - 1e-9
    fv[:] = edge.astype(np.int32)
    obj = scene.objects().create(f"sheet_{kind}")
    slot = obj.geometries().create(mesh)[0]
    SHEETS.append({"kind": kind, "cx": cx, "slot": slot, "gid": obj.geometries().ids()[0],
                   "V0": V0, "free": ~edge,
                   # centre line = the row of vertices nearest z = 0 (where the punch lands)
                   "center": np.abs(V0[:, 2]) < (SHEET / (RES - 1)) * 0.51})

# --- punch: a long rigid bar spanning all three sheets ----------------------
punch_pre = r6.transform4(scale=(2 * PUNCH_HALF_X, 2 * PUNCH_HALF_H, 0.12))
punch_mesh = SimplicialComplexIO(punch_pre).read(f"{TETMESH}/cube.msh")
label_surface(punch_mesh)
label_triangle_orient(punch_mesh)
punch_mesh = flip_inward_triangles(punch_mesh)
abd.apply_to(punch_mesh, 2.0e7)
stc.apply_to(punch_mesh, np.array([3600.0, 120.0], dtype=np.float64))
elem_tool.apply_to(punch_mesh)
punch_mesh.instances().resize(1)
r6.set_instance_transform(punch_mesh, 0, r6.transform4(translate=(0.0, PUNCH_Y_TOP, 0.0)))
punch_obj = scene.objects().create("punch")
punch_slot = punch_obj.geometries().create(punch_mesh)[0]


def punch_height(frame: int) -> float:
    def smooth(a, b, t):
        t = min(max(t, 0.0), 1.0)
        return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * t))
    f = frame
    if f < N_PRESS:
        return smooth(PUNCH_Y_TOP, PUNCH_Y_BOTTOM, f / max(N_PRESS - 1, 1))
    f -= N_PRESS
    if f < N_HOLD:
        return PUNCH_Y_BOTTOM
    f -= N_HOLD
    if f < N_LIFT:
        return smooth(PUNCH_Y_BOTTOM, PUNCH_Y_TOP, f / max(N_LIFT - 1, 1))
    return PUNCH_Y_TOP


def phase(frame: int) -> str:
    if frame < N_PRESS:
        return "press"
    if frame < N_PRESS + N_HOLD:
        return "hold"
    if frame < N_PRESS + N_HOLD + N_LIFT:
        return "lift"
    return "settle"


def animate_punch(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    view(geo.instances().find(builtin.is_constrained))[0] = 1
    T = Matrix4x4.Identity()
    T[1, 3] = punch_height(info.frame())
    view(geo.instances().find(builtin.aim_transform))[0] = T


scene.animator().insert(punch_obj, animate_punch)

ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(-0.30))

world.init(scene)
init_sanity = r6.run_sanity_check(world) if not args.no_sanity else {"skipped": True}
print("SANITY(init):", init_sanity.get("result"), "clean=", init_sanity.get("clean"), flush=True)
if not world.is_valid():
    raise SystemExit("world invalid after init -- sanity check failed")

capture = r6.Capture(args.capture, scene, up_dir="y_up", ground_height=-0.30,
                     stride=args.capture_stride, obj_dir=args.obj)
capture.bind([(s["kind"], s["slot"]) for s in SHEETS] + [("punch", punch_slot)])

runner = r6.Runner(world, engine, scene, args, SCENE_NAME)


def sheet_state(s):
    P = np.asarray(s["slot"].geometry().positions().view()).reshape(-1, 3)
    cl = P[s["center"], 1]
    return {
        "min_y": float(P[:, 1].min()),
        "center_min_y": float(cl.min()),
        "center_mean_y": float(cl.mean()),
        "free_mean_y": float(P[s["free"], 1].mean()),
        "finite": bool(np.all(np.isfinite(P))),
    }


def measure():
    out = {"frame": world.frame(), "phase": phase(world.frame() - 1),
           "punch_y": float(np.asarray(punch_slot.geometry().transforms().view()[0]).reshape(4, 4)[1, 3])}
    for s in SHEETS:
        st = sheet_state(s)
        for k, v in st.items():
            out[f"{s['kind']}_{k}"] = v
    return out


capture.write_frame(0, force=True)
for i in range(N):
    ms = runner.step(on_frame=measure)
    capture.write_frame(i + 1)
    if i % 10 == 0 or i == N - 1:
        t = runner.trace[-1]
        st = runner.frame_stats[-1]
        print(f"frame {i:4d} {ms:7.1f}ms {t['phase']:7s} newton={st['newton_iterations']:2d} "
              f"pcg={st['linear_solver_iterations']:5d} ls={st['line_search_trials']:2d} "
              f"conv={st['converged']} punch={t['punch_y']:.3f} "
              f"crease(el/str/sts)={t['elastic_center_min_y']:.4f}/"
              f"{t['strain_plastic_center_min_y']:.4f}/{t['stress_plastic_center_min_y']:.4f}",
              flush=True)

final_sanity = r6.run_sanity_check(world) if not args.no_sanity else {"skipped": True}
capture.finish(runner.frame_ms)

last = runner.trace[-1]
obs = {"frames": N, "sheet_rest_y": SHEET_Y, "punch_bottom": PUNCH_Y_BOTTOM}
for s in SHEETS:
    k = s["kind"]
    depth = min(t[f"{k}_center_min_y"] for t in runner.trace)
    obs[f"{k}_max_press_depth"] = float(SHEET_Y - depth)
    obs[f"{k}_residual_crease"] = float(SHEET_Y - last[f"{k}_center_min_y"])
    obs[f"{k}_final_center_mean_y"] = float(last[f"{k}_center_mean_y"])
    obs[f"{k}_final_free_mean_y"] = float(last[f"{k}_free_mean_y"])
obs["all_finite"] = bool(all(t[f"{s['kind']}_finite"] for t in runner.trace for s in SHEETS))
obs["punch_returned_to"] = last["punch_y"]

r6.emit(runner, obs, {"sanity": {"init": init_sanity, "final": final_sanity},
                      "trace": runner.trace,
                      "config": {"dt": DT, "res": RES, "sheet": SHEET,
                                 "bending_stiffness": BENDING_STIFFNESS,
                                 "yield_stress": YIELD_STRESS, "yield_strain": YIELD_STRAIN}})
