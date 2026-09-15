"""Example 102 -- "Catch Net".

A square cloth net is slung between four Kirchhoff rods (elastic cables). The
cables are *stitched* to the net corners with SoftVertexStitch inter-primitive
constraints; their top ends are pinned. Three soft StableNeoHookean FEM bodies
and a cloud of Particle points are dropped into the net, which has to arrest
them before they reach the ground.

This is the scene that stresses the FEM side of the solver, so it runs with the
MAS (multilevel additive Schwarz) preconditioner:
    config["linear_system"]["fem_preconditioner"] = "mas"
and it is the only scene here whose degrees of freedom are essentially all
finite-element.

Feature mix: Kirchhoff rods (HookeanSpring stretch + KirchhoffRodBending),
Baraff-Witkin strain-limiting cloth + DiscreteShellBending, SoftVertexStitch,
StableNeoHookean solids, Particle point cloud, implicit-ground contact,
friction, MAS preconditioner, vertex pinning.

Physical idea: an energy budget. All the kinetic energy the payload picks up
falling has to go somewhere -- into cable stretch, net bending, and eventually
friction/damping. The observable is that everything comes to rest *in* the net,
above the floor, with the kinetic energy decaying to ~0.

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

from uipc import Logger, view
import uipc.builtin as builtin
from uipc.core import Engine, Scene, World
from uipc.geometry import (SimplicialComplexIO, ground, linemesh, pointcloud,
                           label_surface, label_triangle_orient, flip_inward_triangles)
from uipc.constitution import (DiscreteShellBending, ElasticModuli, ElasticModuli2D,
                               HookeanSpring, KirchhoffRodBending, Particle,
                               SoftVertexStitch, StableNeoHookean,
                               StrainLimitingBaraffWitkinShell)

SCENE_NAME = "102_net"
TETMESH = AssetDir.tetmesh_path()
TRIMESH = AssetDir.trimesh_path()
DT = 0.01
NET_HALF = 0.80           # net spans [-0.8, 0.8]^2 in x,z
NET_Y = 1.00
ANCHOR_Y = 2.20
ANCHOR_OUT = 1.35         # anchors sit outside the net footprint
ROD_SEGMENTS = 10
GROUND_Y = 0.0
STITCH_GAP = 0.06        # rod end sits this far above the net corner

ap = r6.build_argparser(SCENE_NAME, 250)
args = ap.parse_args()
r6.configure_benchmark_timers()
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Error")))

workspace = args.workspace or AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = DT
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.01
config["linear_system"]["fem_preconditioner"] = os.environ.get("FEM_PRECOND", "mas")
config["linear_system"]["tol_rate"] = float(os.environ.get("TOLRATE", 1e-4))
scene = Scene(config)

ct = scene.contact_tabular()
ct.default_model(0.35, 1.0e9)
elem_net = ct.create("net")
elem_body = ct.create("body")
elem_rod = ct.create("rod")
ct.insert(elem_net, elem_body, 0.45, 1.0e9)
ct.insert(elem_net, elem_net, 0.40, 1.0e9)
ct.insert(elem_body, elem_body, 0.40, 1.0e9)
ct.insert(elem_rod, elem_net, 0.30, 1.0e9)
ct.insert(elem_rod, elem_body, 0.30, 1.0e9)

spring = HookeanSpring()
rod_bend = KirchhoffRodBending()
bw = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()
snh = StableNeoHookean()
particle = Particle()
stitcher = SoftVertexStitch()

# ---------------------------------------------------------------------------
# the net (cloth grid, 40x40 -> 1600 vertices)
# ---------------------------------------------------------------------------
net_pre = r6.transform4(translate=(0.0, NET_Y, 0.0), scale=(2 * NET_HALF, 1.0, 2 * NET_HALF))
net_mesh = SimplicialComplexIO(net_pre).read(f"{TRIMESH}/grid40x40.obj")
label_surface(net_mesh)
net_V0 = np.asarray(net_mesh.positions().view()).reshape(-1, 3).copy()
bw.apply_to(net_mesh,
            stretch_moduli=ElasticModuli2D.youngs_poisson(2.0e6, 0.35),
            shear_moduli=ElasticModuli2D.youngs_poisson(2.0e3, 0.35),
            mass_density=300.0, thickness=3e-3, strain_rate=100.0)
dsb.apply_to(net_mesh, 2.0e2)
elem_net.apply_to(net_mesh)
net_obj = scene.objects().create("net")
net_slot = net_obj.geometries().create(net_mesh)[0]
net_gid = net_obj.geometries().ids()[0]

# net corner vertex indices (nearest vertex to each corner of the footprint)
corners = []
for sx in (-1, 1):
    for sz in (-1, 1):
        target = np.array([sx * NET_HALF, NET_Y, sz * NET_HALF])
        corners.append(int(np.argmin(np.linalg.norm(net_V0 - target, axis=1))))

# ---------------------------------------------------------------------------
# four Kirchhoff rods: anchor (pinned, high and outboard) -> net corner
# ---------------------------------------------------------------------------
rod_slots = []
rod_info = []
for k, ci in enumerate(corners):
    corner = net_V0[ci]
    # the rod stops a little above the net corner; the stitch has that gap as
    # its rest length, so nothing starts in contact (surface-distance check)
    end = corner + np.array([0.0, STITCH_GAP, 0.0])
    start = np.array([np.sign(end[0]) * ANCHOR_OUT, ANCHOR_Y, np.sign(end[2]) * ANCHOR_OUT])
    ts = np.linspace(0.0, 1.0, ROD_SEGMENTS + 1)[:, None]
    V = start[None, :] * (1 - ts) + end[None, :] * ts
    E = np.array([[i, i + 1] for i in range(ROD_SEGMENTS)], dtype=np.int32)
    rod = linemesh(V.astype(np.float64), E)
    label_surface(rod)
    spring.apply_to(rod, 4.0e7, 2.0e3, 0.015)       # moduli, density, radius
    rod_bend.apply_to(rod, 5.0e5)
    elem_rod.apply_to(rod)
    fixed = rod.vertices().find(builtin.is_fixed)
    if fixed is None:
        fixed = rod.vertices().create(builtin.is_fixed, 0)
    view(fixed)[0] = 1                               # pin the anchor end
    obj = scene.objects().create(f"cable{k}")
    slot = obj.geometries().create(rod)[0]
    rod_slots.append(slot)
    rod_info.append({"gid": obj.geometries().ids()[0], "rest_len": float(np.linalg.norm(end - start))})

# stitch each rod's free end to the matching net corner
for k, (slot, ci) in enumerate(zip(rod_slots, corners)):
    pairs = np.array([[ROD_SEGMENTS, ci]], dtype=np.int32)
    st = stitcher.create_geometry((slot, net_slot), pairs, 2.0e6, STITCH_GAP)
    scene.objects().create(f"stitch{k}").geometries().create(st)

# ---------------------------------------------------------------------------
# payload: three soft FEM bodies + a particle cloud
# ---------------------------------------------------------------------------
DROPS = [((-0.34, 1.75, -0.30), 0.55, 0.0),
         ((0.36, 2.05, 0.28), 0.50, 1.2),
         ((0.05, 2.45, -0.38), 0.45, 2.4)]
bodies = []
for i, (pos, s, yaw) in enumerate(DROPS):
    pre = r6.transform4(translate=pos, scale=s, rotate=((0, 1, 0), yaw))
    m = SimplicialComplexIO(pre).read(f"{TETMESH}/bunny0.msh")
    label_surface(m)
    label_triangle_orient(m)
    m = flip_inward_triangles(m)
    snh.apply_to(m, ElasticModuli.youngs_poisson(1.5e5, 0.45), 6.0e2)
    elem_body.apply_to(m)
    o = scene.objects().create(f"payload{i}")
    slot = o.geometries().create(m)[0]
    bodies.append({"name": f"payload{i}", "slot": slot, "gid": o.geometries().ids()[0]})

# particle "gravel": exercises the Particle constitution
rng = np.random.default_rng(20260912)
GRID = int(os.environ.get("GRAVEL_GRID", 5))
PART_R = 0.015
step = 0.10
gx = (np.arange(GRID) - (GRID - 1) / 2) * step
gy = 3.05 + np.arange(GRID) * step
pts = np.array([[x, y, z] for y in gy for z in gx for x in gx], dtype=np.float64)
pts += rng.uniform(-0.012, 0.012, pts.shape)     # jitter, still >> 2*PART_R apart
NP = len(pts)
pc = pointcloud(pts)
label_surface(pc)
particle.apply_to(pc, 8.0e2, PART_R)
elem_body.apply_to(pc)
grav_obj = scene.objects().create("gravel")
grav_slot = grav_obj.geometries().create(pc)[0]
bodies.append({"name": "gravel", "slot": grav_slot, "gid": grav_obj.geometries().ids()[0]})

ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(GROUND_Y))

# ---------------------------------------------------------------------------
world.init(scene)
init_sanity = r6.run_sanity_check(world) if not args.no_sanity else {"skipped": True}
print("SANITY(init):", init_sanity.get("result"), "clean=", init_sanity.get("clean"), flush=True)
if not world.is_valid():
    raise SystemExit("world invalid after init -- sanity check failed")

capture = r6.Capture(args.capture, scene, up_dir="y_up", ground_height=GROUND_Y,
                     stride=args.capture_stride, obj_dir=args.obj,
                     radius={f"cable{k}": 0.015 for k in range(4)} | {"gravel": PART_R})
capture.bind([("net", net_slot)] + [(f"cable{k}", s) for k, s in enumerate(rod_slots)]
             + [(b["name"], b["slot"]) for b in bodies])

runner = r6.Runner(world, engine, scene, args, SCENE_NAME)

ALL_SLOTS = [("net", net_slot)] + [(f"cable{k}", s) for k, s in enumerate(rod_slots)] \
            + [(b["name"], b["slot"]) for b in bodies]


# per-vertex masses (volume attribute x the geometry's mass density)
VERTEX_MASS = {}
for _n, _s in ALL_SLOTS:
    _g = _s.geometry()
    _vol = _g.vertices().find(builtin.volume)
    _rho = _g.meta().find(builtin.mass_density)
    if _vol is not None and _rho is not None:
        VERTEX_MASS[_n] = (np.asarray(_vol.view()).reshape(-1).copy()
                           * float(np.asarray(_rho.view()).reshape(-1)[0]))
_PREV = {}


def kinetic_energy():
    """1/2 m v^2 summed over all simulated vertices.

    The frontend `velocity` attribute is not refreshed by `world.retrieve()`,
    so velocities are finite-differenced from the retrieved positions.
    """
    total = 0.0
    for name, slot in ALL_SLOTS:
        P = np.asarray(slot.geometry().positions().view()).reshape(-1, 3)
        prev = _PREV.get(name)
        _PREV[name] = P.copy()
        M = VERTEX_MASS.get(name)
        if prev is None or M is None or prev.shape != P.shape:
            continue
        V = (P - prev) / DT
        total += 0.5 * float(np.sum(M * np.sum(V * V, axis=1)))
    return total


def measure():
    out = {"frame": world.frame()}
    P = np.asarray(net_slot.geometry().positions().view()).reshape(-1, 3)
    out["net_min_y"] = float(P[:, 1].min())
    out["net_center_y"] = float(P[np.argmin(np.linalg.norm(net_V0[:, [0, 2]], axis=1)), 1])
    out["net_finite"] = bool(np.all(np.isfinite(P)))
    fem_min = []
    for b in bodies:
        Q = np.asarray(b["slot"].geometry().positions().view()).reshape(-1, 3)
        out[f"c_{b['name']}"] = Q.mean(axis=0).tolist()
        out[f"min_y_{b['name']}"] = float(Q[:, 1].min())
        if b["name"] == "gravel":
            out["gravel_min_y"] = float(Q[:, 1].min())
            out["gravel_below_net"] = int(np.sum(Q[:, 1] < 0.25))
        else:
            fem_min.append(float(Q[:, 1].min()))
        if not np.all(np.isfinite(Q)):
            out["net_finite"] = False
    out["payload_min_y"] = float(min(fem_min))
    stretch = []
    for k, s in enumerate(rod_slots):
        R = np.asarray(s.geometry().positions().view()).reshape(-1, 3)
        L = float(np.sum(np.linalg.norm(np.diff(R, axis=0), axis=1)))
        stretch.append(L / rod_info[k]["rest_len"])
    out["rod_max_stretch"] = float(max(stretch))
    out["kinetic_energy"] = kinetic_energy()
    return out


capture.write_frame(0, force=True)
for i in range(args.frames):
    ms = runner.step(on_frame=measure)
    capture.write_frame(i + 1)
    if i % 10 == 0 or i == args.frames - 1:
        t = runner.trace[-1]
        st = runner.frame_stats[-1]
        print(f"frame {i:4d} {ms:7.1f}ms newton={st['newton_iterations']:2d} "
              f"pcg={st['linear_solver_iterations']:5d} ls={st['line_search_trials']:2d} "
              f"conv={st['converged']} net_min_y={t['net_min_y']:.4f} "
              f"fem_min_y={t['payload_min_y']:.4f} grav_min_y={t['gravel_min_y']:.4f} "
              f"stretch={t['rod_max_stretch']:.4f} "
              f"KE={t['kinetic_energy']:.3f}", flush=True)

final_sanity = r6.run_sanity_check(world) if not args.no_sanity else {"skipped": True}
capture.finish(runner.frame_ms)

ke = np.array([t["kinetic_energy"] for t in runner.trace])
last = runner.trace[-1]
obs = {
    "frames": args.frames,
    "preconditioner": config["linear_system"]["fem_preconditioner"],
    "net_rest_y": NET_Y,
    "net_final_min_y": last["net_min_y"],
    "net_final_center_y": last["net_center_y"],
    "net_max_sag": float(NET_Y - min(t["net_min_y"] for t in runner.trace)),
    "payload_final_min_y": last["payload_min_y"],
    "payload_min_y_ever": float(min(t["payload_min_y"] for t in runner.trace)),
    "gravel_final_min_y": last["gravel_min_y"],
    "gravel_min_y_ever": float(min(t["gravel_min_y"] for t in runner.trace)),
    "gravel_below_net_final": int(last["gravel_below_net"]),
    "n_particles": NP,
    "ground_y": GROUND_Y,
    "rod_max_stretch_ever": float(max(t["rod_max_stretch"] for t in runner.trace)),
    "rod_final_stretch": last["rod_max_stretch"],
    "ke_peak": float(ke.max()),
    "ke_final": float(ke[-1]),
    "ke_final_over_peak": float(ke[-1] / max(ke.max(), 1e-30)),
    "all_finite": bool(all(t["net_finite"] for t in runner.trace)),
    "final_centroids": {k[2:]: v for k, v in last.items() if k.startswith("c_")},
}
r6.emit(runner, obs, {"sanity": {"init": init_sanity, "final": final_sanity},
                      "trace": runner.trace,
                      "config": {"dt": DT, "net_half": NET_HALF, "rod_segments": ROD_SEGMENTS,
                                 "n_particles": NP}})
