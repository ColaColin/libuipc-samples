"""Example 100 -- "Paddle Mixer".

A motor-driven affine-body paddle, hinged to a fixed post by a *revolute joint*
and turned by a *driving revolute joint* (angle motor), sweeps through a square
bin filled with a deliberately heterogeneous pile:

  * three StableNeoHookean FEM bunnies (soft, self-colliding)
  * one ARAP FEM cube-lattice (the rarely used as-rigid-as-possible solid)
  * three affine-body rigid links (OrthoPotential ABD)
  * one Baraff-Witkin strain-limiting cloth rag with DiscreteShellBending
  * a trailing flail: a second ABD bar hung off the paddle by a *second*
    revolute joint, so the scene contains a genuine two-link joint chain

The bin is made of implicit half-planes (a ground plus four walls) so nothing
can leave the domain -- which also makes the half-plane sanity checker
meaningful.

Physical idea: a single kinematic degree of freedom (the commanded paddle
angle) drives a pile of very different materials through frictional contact.
The observable is that the paddle *tracks its commanded angle* while doing the
work, and that every body stays inside the bin.

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

from uipc import Animation, Logger, Matrix4x4, Transform, Vector2, Vector3, view
import uipc.builtin as builtin
from uipc.core import Engine, Scene, World
from uipc.geometry import (SimplicialComplex, SimplicialComplexIO, ground, halfplane,
                           label_surface, label_triangle_orient, flip_inward_triangles)
from uipc.constitution import (AffineBodyConstitution, AffineBodyRevoluteJoint,
                               AffineBodyDrivingRevoluteJoint, ARAP,
                               DiscreteShellBending, ElasticModuli, ElasticModuli2D,
                               StableNeoHookean, StrainLimitingBaraffWitkinShell)
from uipc.unit import MPa, GPa

SCENE_NAME = "100_mixer"
TETMESH = AssetDir.tetmesh_path()
TRIMESH = AssetDir.trimesh_path()
MOTOR_RAD_PER_S = 1.1          # commanded paddle angular velocity
DT = 0.02

ap = r6.build_argparser(SCENE_NAME, 150)
args = ap.parse_args()

r6.configure_benchmark_timers()
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Error")))

workspace = args.workspace or AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

# ---------------------------------------------------------------------------
config = Scene.default_config()
config["dt"] = DT
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.01
# A thin, light shell coupled to stiff affine bodies needs a tighter linear
# solve than the 1e-3 default: the cloth's gravity residual is otherwise far
# below the *relative* PCG stopping criterion set by the stiff bodies, and the
# cloth simply never moves (it hangs in mid-air while every other body
# behaves; bisected to a minimal cloth+ABD+joint case, reproducing on
# upstream, before it was understood).  The heavier 600 kg/m^3 x 4 mm
# tarpaulin rag below is the other half of that fix.
config["linear_system"]["tol_rate"] = 1e-5
scene = Scene(config)

# --- contact tabular: several elements with different friction pairs --------
ct = scene.contact_tabular()
ct.default_model(0.3, 1.0 * GPa)
elem_default = ct.default_element()
elem_metal = ct.create("metal")      # paddle / links: slick
elem_soft = ct.create("soft")        # FEM bodies: grippy
elem_cloth = ct.create("cloth")
ct.insert(elem_metal, elem_metal, 0.10, 1.0 * GPa)
ct.insert(elem_metal, elem_soft, 0.35, 1.0 * GPa)
ct.insert(elem_metal, elem_cloth, 0.25, 1.0 * GPa)
ct.insert(elem_soft, elem_soft, 0.50, 1.0 * GPa)
ct.insert(elem_soft, elem_cloth, 0.45, 1.0 * GPa)
ct.insert(elem_cloth, elem_cloth, 0.40, 1.0 * GPa)

abd = AffineBodyConstitution()
snh = StableNeoHookean()
arap = ARAP()
bw = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()


def process_tet(mesh: SimplicialComplex) -> SimplicialComplex:
    label_surface(mesh)
    label_triangle_orient(mesh)
    return flip_inward_triangles(mesh)


def read_tet(path, scale=1.0, translate=(0, 0, 0), rotate=None):
    pre = r6.transform4(translate=translate, scale=scale, rotate=rotate)
    io = SimplicialComplexIO(pre)
    return process_tet(io.read(path))


def box_mesh(size, center=(0, 0, 0)):
    """A single-body box built from the 5-tet unit cube asset."""
    pre = r6.transform4(translate=center, scale=np.asarray(size, dtype=float))
    return process_tet(SimplicialComplexIO(pre).read(f"{TETMESH}/cube.msh"))


# ---------------------------------------------------------------------------
# the bin: ground + four infinite walls (implicit half-planes)
# ---------------------------------------------------------------------------
BIN = 1.60
bin_obj = scene.objects().create("bin")
bin_obj.geometries().create(ground(0.0))
for P, N in (((BIN, 0, 0), (-1, 0, 0)), ((-BIN, 0, 0), (1, 0, 0)),
             ((0, 0, BIN), (0, 0, -1)), ((0, 0, -BIN), (0, 0, 1))):
    bin_obj.geometries().create(halfplane(np.array(P, dtype=float), np.array(N, dtype=float)))

# ---------------------------------------------------------------------------
# fixed post + paddle, joined by a driven revolute joint about the vertical axis
# ---------------------------------------------------------------------------
post_mesh = box_mesh((0.18, 0.85, 0.18), (0.0, 0.975, 0.0))
abd.apply_to(post_mesh, 100.0 * MPa)
elem_metal.apply_to(post_mesh)
post_mesh.instances().resize(1)
view(post_mesh.instances().find(builtin.is_fixed))[0] = 1
post_obj = scene.objects().create("post")
post_slot = post_obj.geometries().create(post_mesh)[0]

# paddle: a flat bar, long in x, sweeping horizontally just above the floor
paddle_mesh = box_mesh((2.1, 0.40, 0.10), (0.0, 0.30, 0.0))
abd.apply_to(paddle_mesh, 80.0 * MPa)
elem_metal.apply_to(paddle_mesh)
paddle_mesh.instances().resize(1)
view(paddle_mesh.instances().find(builtin.is_fixed))[0] = 0
paddle_obj = scene.objects().create("paddle")
paddle_slot = paddle_obj.geometries().create(paddle_mesh)[0]

# flail: a second bar hung off the paddle tip by a passive revolute joint
flail_mesh = box_mesh((0.12, 0.42, 0.55), (0.0, 0.0, 0.0))
abd.apply_to(flail_mesh, 60.0 * MPa)
elem_metal.apply_to(flail_mesh)
flail_mesh.instances().resize(1)
r6.set_instance_transform(flail_mesh, 0, r6.transform4(translate=(1.22, 0.30, 0.0)))
view(flail_mesh.instances().find(builtin.is_fixed))[0] = 0
flail_obj = scene.objects().create("flail")
flail_slot = flail_obj.geometries().create(flail_mesh)[0]

# joints: ONE joint geometry with two edges (the backend pools all revolute
# joints, and the driving constitution must cover the same edge set).
#   edge 0 -- post  <-> paddle : driven (angle motor)
#   edge 1 -- paddle<-> flail  : passive
revolute = AffineBodyRevoluteJoint()
joint_mesh = revolute.create_geometry(
    np.array([[0.0, 0.10, 0.0], [1.10, 0.10, 0.0]], dtype=np.float64),
    np.array([[0.0, 1.10, 0.0], [1.10, 1.10, 0.0]], dtype=np.float64),
    [post_slot, paddle_slot], np.array([0, 0], dtype=np.int32),
    [paddle_slot, flail_slot], np.array([0, 0], dtype=np.int32),
    np.array([200.0, 200.0], dtype=np.float64))
driving = AffineBodyDrivingRevoluteJoint()
driving.apply_to(joint_mesh, np.array([200.0, 0.0], dtype=np.float64))
drive_joint_obj = scene.objects().create("hinges")
drive_joint_obj.geometries().create(joint_mesh)


def animate_drive(info: Animation.UpdateInfo):
    for geo_slot in info.geo_slots():
        geo = geo_slot.geometry()
        angles = view(geo.edges().find("angle"))
        ic = view(geo.edges().find("driving/is_constrained"))
        ic[:] = 0
        ic[0] = 1                     # edge 0 driven, edge 1 free to swing
        aim = geo.edges().find("aim_angle")
        if aim is not None:
            av = view(aim)
            av[0] = angles[0] + info.dt() * MOTOR_RAD_PER_S


scene.animator().insert(drive_joint_obj, animate_drive)

# ---------------------------------------------------------------------------
# the pile
# ---------------------------------------------------------------------------
pile = []

BUNNY_POSE = [((0.55, 0.30, 0.85), 0.0), ((-0.60, 0.30, 0.85), 0.0), ((0.00, 0.30, -0.85), 0.0)]
for i, (pos, yaw) in enumerate(BUNNY_POSE):
    m = read_tet(f"{TETMESH}/bunny0.msh", scale=0.40, translate=pos,
                 rotate=((0, 1, 0), yaw))
    snh.apply_to(m, ElasticModuli.youngs_poisson(2.0e5, 0.45), 1.0e3)
    elem_soft.apply_to(m)
    o = scene.objects().create(f"bunny{i}")
    pile.append((f"bunny{i}", o.geometries().create(m)[0], o.geometries().ids()[0]))

# ARAP block: the as-rigid-as-possible solid, a stiff but still deformable body
arap_mesh = read_tet(f"{TETMESH}/cylinder_hole.msh", scale=0.55, translate=(-0.85, 0.30, -0.80),
                     rotate=((1, 0, 0), math.pi / 2))
arap.apply_to(arap_mesh, 5.0e5, 1.0e3)
elem_soft.apply_to(arap_mesh)
arap_obj = scene.objects().create("arap_ring")
pile.append(("arap_ring", arap_obj.geometries().create(arap_mesh)[0], arap_obj.geometries().ids()[0]))

# rigid ABD links
LINK_POSE = [((1.10, 0.36, 0.75), 0.4), ((-1.15, 0.36, 0.50), 1.9), ((0.55, 0.36, -1.05), 0.9)]
for i, (pos, yaw) in enumerate(LINK_POSE):
    m = read_tet(f"{TETMESH}/link.msh", scale=0.45, translate=pos, rotate=((0, 1, 0), yaw))
    abd.apply_to(m, 50.0 * MPa)
    elem_metal.apply_to(m)
    m.instances().resize(1)
    o = scene.objects().create(f"link{i}")
    pile.append((f"link{i}", o.geometries().create(m)[0], o.geometries().ids()[0]))

# cloth rag (Baraff-Witkin strain limiting + discrete shell bending)
rag_pre = r6.transform4(translate=(0.0, 0.85, 0.95), scale=(0.9, 1.0, 0.9))
rag_mesh = SimplicialComplexIO(rag_pre).read(f"{TRIMESH}/grid20x20.obj")
label_surface(rag_mesh)
bw.apply_to(rag_mesh,
            stretch_moduli=ElasticModuli2D.youngs_poisson(2.0e5, 0.30),
            shear_moduli=ElasticModuli2D.youngs_poisson(1.0e3, 0.30),
            mass_density=600.0, thickness=4e-3, strain_rate=100.0)
dsb.apply_to(rag_mesh, 1.0e3)
elem_cloth.apply_to(rag_mesh)
rag_obj = scene.objects().create("rag")
pile.append(("rag", rag_obj.geometries().create(rag_mesh)[0], rag_obj.geometries().ids()[0]))

# ---------------------------------------------------------------------------
world.init(scene)
init_sanity = r6.run_sanity_check(world, "post-init") if not args.no_sanity else {"skipped": True}
print("SANITY(init):", init_sanity.get("result"), "clean=", init_sanity.get("clean"), flush=True)
if not world.is_valid():
    raise SystemExit("world invalid after init -- sanity check failed")

paddle_id = paddle_obj.geometries().ids()[0]
flail_id = flail_obj.geometries().ids()[0]
joint_geo_id = drive_joint_obj.geometries().ids()[0]

capture = r6.Capture(args.capture, scene, up_dir="y_up", ground_height=0.0,
                     stride=args.capture_stride, obj_dir=args.obj)
capture.bind([(name, slot) for name, slot, _ in pile]
             + [("paddle", paddle_slot), ("flail", flail_slot), ("post", post_slot)])

runner = r6.Runner(world, engine, scene, args, SCENE_NAME)


def body_angle_y(slot):
    """Signed rotation about +Y read straight off the affine-body transform."""
    A = np.asarray(slot.geometry().transforms().view()[0]).reshape(4, 4)
    return math.atan2(-A[2, 0], A[0, 0])


def paddle_angle():
    return body_angle_y(paddle_slot)


def measure():
    stats = {}
    allP = []
    for name, _slot, gid in pile:
        P = r6.world_positions(scene, gid)
        allP.append(P)
        stats[f"c_{name}"] = P.mean(axis=0).tolist()
    A = np.vstack(allP)
    stats["min_y"] = float(A[:, 1].min())
    stats["max_abs_x"] = float(np.abs(A[:, 0]).max())
    stats["max_abs_z"] = float(np.abs(A[:, 2]).max())
    stats["finite"] = bool(np.all(np.isfinite(A)))
    stats["paddle_angle"] = paddle_angle()
    stats["flail_angle"] = body_angle_y(flail_slot)
    stats["frame"] = world.frame()
    return stats


base = measure()
capture.write_frame(0, force=True)

for i in range(args.frames):
    ms = runner.step(on_frame=measure)
    capture.write_frame(i + 1)
    if i % 10 == 0 or i == args.frames - 1:
        t = runner.trace[-1]
        s = runner.frame_stats[-1]
        print(f"frame {i:4d} {ms:7.1f}ms  newton={s['newton_iterations']:2d} "
              f"pcg={s['linear_solver_iterations']:5d} ls={s['line_search_trials']:2d} "
              f"conv={s['converged']}  min_y={t['min_y']:+.4f} "
              f"paddle={math.degrees(t['paddle_angle']):+7.2f}deg", flush=True)

final_sanity = r6.run_sanity_check(world, "final") if not args.no_sanity else {"skipped": True}
capture.finish(runner.frame_ms)

# --- observables ------------------------------------------------------------
ang = np.unwrap([t["paddle_angle"] for t in runner.trace])
ang0 = np.unwrap([base["paddle_angle"], ang[0]])[0]
commanded = MOTOR_RAD_PER_S * DT * np.arange(1, len(ang) + 1)
tracking_err = ang - ang0 - commanded
last = runner.trace[-1]
observables = {
    "frames": args.frames,
    "paddle_total_turn_deg": float(math.degrees(ang[-1] - ang0)),
    "paddle_commanded_turn_deg": float(math.degrees(commanded[-1])),
    "paddle_tracking_err_deg_max": float(np.max(np.abs(np.degrees(tracking_err)))),
    "paddle_tracking_err_deg_final": float(math.degrees(tracking_err[-1])),
    "pile_min_y": float(min(t["min_y"] for t in runner.trace)),
    "pile_max_abs_x": float(max(t["max_abs_x"] for t in runner.trace)),
    "pile_max_abs_z": float(max(t["max_abs_z"] for t in runner.trace)),
    "bin_half_width": BIN,
    "all_finite": bool(all(t["finite"] for t in runner.trace)),
    "final_centroids": {k[2:]: v for k, v in last.items() if k.startswith("c_")},
}
extra = {
    "sanity": {"init": init_sanity, "final": final_sanity},
    "trace": runner.trace,
    "config": {"dt": DT, "motor_rad_per_s": MOTOR_RAD_PER_S, "bin": BIN},
}
r6.emit(runner, observables, extra)
