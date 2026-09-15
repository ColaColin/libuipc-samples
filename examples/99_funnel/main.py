"""Example 99 -- "Cloth press through a funnel" (a soft mangle).

Five sheets of the tumbler's calibrated cloth hang as a stack, their lower
ends already inside the narrowing gap between two *soft* FEM rollers
(StableNeoHookean, structured hex->tet cylinders).  Each roller is driven
through its core by a SoftPositionConstraint whose aim positions are the
rest positions rotated about the roller axis (the twisting-bar idiom): the
two rollers close on the stack over the first half second and counter-rotate
so their surfaces run *down* through the nip.  The stack's top rows are held
by a second soft position constraint until the rollers have gripped, then
released.  The rollers deform around the stack, squeeze the layers together
and pull them through; below the nip the cloth piles up on the floor in a
dense multi-layer heap.

Observables: cloth passes through -- the fraction of cloth area below the
nip grows to most of the stack; the roller cores track their aim; the layers
stay finite, un-inverted and inside the domain.

Headless:  python main.py [FRAMES] [--edge-len M] [--capture DIR] [--result JSON]
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
from r6_common import G

from uipc import Animation, Logger, builtin, view
from uipc.core import Engine, Scene, World
from uipc.geometry import ground, halfplane
from uipc.constitution import ElasticModuli, SoftPositionConstraint, StableNeoHookean

SCENE_NAME = "99_funnel"
DT = r6.DT
N_SHEETS = 5
SHEET_L, SHEET_W = 1.0, 0.5          # hanging length (y), width (z)
ROLL_R, ROLL_L = 0.15, 0.6
ROLL_E, ROLL_NU, ROLL_RHO = 2.0e5, 0.45, 1000.0
CORE_FRAC = 0.6                       # constrained core radius / R
SURFACE_SPEED = 0.30                  # m/s
OMEGA = SURFACE_SPEED / ROLL_R
CLOSE_T0, CLOSE_T1 = 0.10, 0.50       # gap closing window (s)
SPIN_T0 = 0.30                        # rollers start turning
RELEASE_T = 0.60                      # top rows released
INSERT = 0.05                         # how far the sheets start inside the nip
GROUND_Y = -0.70
WALL_Z = 0.60

ap = r6.build_argparser(SCENE_NAME, 300)
r6.add_r6_args(ap, 0.012)
ap.add_argument("--roller-cells", type=int, default=8)
args = ap.parse_args()
EDGE_LEN = args.edge_len

r6.configure_benchmark_timers()
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Error")))
workspace = args.workspace or AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

# --------------------------------------------------------------------------
V0, F0 = G.build_sheet(SHEET_L, SHEET_W, EDGE_LEN)        # x = L, y = W
st = G.mesh_stats(V0, F0)
CLOTH_R, D_HAT = r6.contact_resolution(st["min_tri_height"])
SPACING = r6.layer_gap(CLOTH_R, D_HAT)
STACK = (N_SHEETS - 1) * SPACING
GAP0 = STACK + 2 * SPACING + 0.004
GAP1 = 0.40 * GAP0
CX0 = ROLL_R + GAP0 / 2                # roller centre |x| at t=0
# The whole rig is yawed 90 deg about y so a fixed orbit (camera on the +x
# side, 24 deg elevation, +-35 deg azimuth) looks along the roller axis and
# down into the nip line: the squeezed layer stack is seen in cross-section
# between the two roller ends, the pile below it.
# Physics is written in the local frame (roller axis = z) and mapped with RY.
YAW = math.radians(90.0)
RY = np.array([[math.cos(YAW), 0.0, math.sin(YAW)], [0.0, 1.0, 0.0],
               [-math.sin(YAW), 0.0, math.cos(YAW)]])


def to_world(P):
    return np.asarray(P, dtype=np.float64) @ RY.T


sheets = []
for i in range(N_SHEETS):
    x = (i - (N_SHEETS - 1) / 2) * SPACING
    Vs = np.stack([np.full(len(V0), x), V0[:, 0] + SHEET_L / 2 - INSERT, V0[:, 1]], axis=1)
    sheets.append(to_world(Vs))

config = r6.common_config(D_HAT, tol_rate=args.tol_rate or 1e-4)
scene = Scene(config)
ct = scene.contact_tabular()
ct.default_model(0.4, r6.CONTACT_RESISTANCE)
elem_roll = ct.create("roller")
elem_cloth = ct.create("cloth")
ct.insert(elem_roll, elem_cloth, 0.6, r6.CONTACT_RESISTANCE)
ct.insert(elem_roll, elem_roll, 0.3, r6.CONTACT_RESISTANCE)
ct.insert(elem_cloth, elem_cloth, 0.3, r6.CONTACT_RESISTANCE)

snh = StableNeoHookean()
spc = SoftPositionConstraint()

# --- rollers ------------------------------------------------------------------
rollers = []
for side in (-1.0, +1.0):
    cx = side * CX0
    V, T = r6.tet_cylinder(ROLL_R, ROLL_L, args.roller_cells, args.roller_cells + 2,
                           axis="z", center=(cx, 0.0, 0.0))
    m = r6.make_tetmesh(to_world(V), T)
    snh.apply_to(m, ElasticModuli.youngs_poisson(ROLL_E, ROLL_NU), ROLL_RHO)
    spc.apply_to(m, 100.0)
    elem_roll.apply_to(m)
    core = np.hypot(V[:, 0] - cx, V[:, 1]) <= CORE_FRAC * ROLL_R + 1e-9
    name = "roller_L" if side < 0 else "roller_R"
    o = scene.objects().create(name)
    slot, _ = o.geometries().create(m)
    rollers.append(dict(name=name, side=side, cx0=cx, V=V, T=T, core=core, obj=o,
                        slot=slot, gid=o.geometries().ids()[0]))


def roller_pose(side, t):
    """(centre x, rotation angle) commanded at time t."""
    s = min(1.0, max(0.0, (t - CLOSE_T0) / (CLOSE_T1 - CLOSE_T0)))
    gap = GAP0 + (GAP1 - GAP0) * s
    cx = side * (ROLL_R + gap / 2)
    # surfaces at the nip move -y: left roller omega < 0, right roller omega > 0
    theta = side * OMEGA * max(0.0, t - SPIN_T0)
    return cx, theta


def make_roller_animator(rd):
    rest = rd["V"]
    core = rd["core"]
    rel = rest[core] - np.array([rd["cx0"], 0.0, 0.0])

    def animate(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        t = info.frame() * info.dt()
        cx, th = roller_pose(rd["side"], t)
        c, s = math.cos(th), math.sin(th)
        aim = rest.copy()
        aim[core, 0] = c * rel[:, 0] - s * rel[:, 1] + cx
        aim[core, 1] = s * rel[:, 0] + c * rel[:, 1]
        view(geo.vertices().find(builtin.aim_position))[:] = to_world(aim).reshape(-1, 3, 1)
        view(geo.vertices().find(builtin.is_constrained))[:] = core.astype(np.int32)
    return animate


for rd in rollers:
    scene.animator().insert(rd["obj"], make_roller_animator(rd))

# --- sheets ----------------------------------------------------------------------
cloth = r6.ClothMaterial(CLOTH_R, elem_cloth)
TOP = np.flatnonzero(V0[:, 0] > V0[:, 0].max() - 1e-6)
sheet_recs = []
for i, Vs in enumerate(sheets):
    m = cloth.make(Vs, F0)
    spc.apply_to(m, 100.0)
    name = f"sheet{i}"
    o = scene.objects().create(name)
    slot, _ = o.geometries().create(m)
    sheet_recs.append(dict(name=name, V=Vs, obj=o, slot=slot, gid=o.geometries().ids()[0]))


def make_sheet_animator(rec):
    rest = rec["V"]
    hold = np.zeros(len(rest), dtype=np.int32)
    hold[TOP] = 1

    def animate(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        t = info.frame() * info.dt()
        view(geo.vertices().find(builtin.aim_position))[:] = rest.reshape(-1, 3, 1)
        ic = view(geo.vertices().find(builtin.is_constrained))
        ic[:] = hold if t < RELEASE_T else 0
    return animate


for rec in sheet_recs:
    scene.animator().insert(rec["obj"], make_sheet_animator(rec))

# --- domain ---------------------------------------------------------------------------
dom = scene.objects().create("domain")
dom.geometries().create(ground(GROUND_Y))
for P, N in (((0, 0, WALL_Z), (0, 0, -1)), ((0, 0, -WALL_Z), (0, 0, 1))):
    dom.geometries().create(halfplane(to_world(P), to_world(N)))

# --------------------------------------------------------------------------
world.init(scene)
init_sanity = r6.run_sanity_check(world, "post-init") if not args.no_sanity else {"skipped": True}
print("SANITY(init):", init_sanity.get("result"), "clean=", init_sanity.get("clean"), flush=True)
if not world.is_valid():
    raise SystemExit("world invalid after init -- sanity check failed")

N_TRIS = N_SHEETS * len(F0)
N_TETS = sum(len(r["T"]) for r in rollers)
print(f"{SCENE_NAME}: {N_SHEETS} sheets x {len(F0)} tris = {N_TRIS} tris, rollers {N_TETS} tets, "
      f"edge_len={EDGE_LEN*1e3:.1f}mm h_min={st['min_tri_height']*1e3:.2f}mm r={CLOTH_R*1e3:.2f}mm "
      f"d_hat={D_HAT*1e3:.2f}mm spacing={SPACING*1e3:.1f}mm gap {GAP0*1e3:.1f}->{GAP1*1e3:.1f}mm "
      f"frames={args.frames}", flush=True)

capture = r6.Capture(args.capture, scene, up_dir="y_up", ground_height=GROUND_Y,
                     stride=args.capture_stride, obj_dir=args.obj)
capture.bind([(r["name"], r["slot"]) for r in sheet_recs] + [(r["name"], r["slot"]) for r in rollers])
runner = r6.Runner(world, engine, scene, args, SCENE_NAME)
obs = r6.ClothObserver(scene, [(r["name"], r["gid"], r["V"], F0) for r in sheet_recs],
                       DT, CLOTH_R, D_HAT, bounds={"radius_max": 1.0, "y_max": SHEET_L + 0.2},
                       ground_y=GROUND_Y)


def roller_track_err():
    worst = 0.0
    t = world.frame() * DT
    for rd in rollers:
        P = r6.world_positions(scene, rd["gid"])
        cx, th = roller_pose(rd["side"], t)
        c, s = math.cos(th), math.sin(th)
        rel = rd["V"][rd["core"]] - np.array([rd["cx0"], 0.0, 0.0])
        aim = np.stack([c * rel[:, 0] - s * rel[:, 1] + cx, s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], 1)
        worst = max(worst, float(np.linalg.norm(P[rd["core"]] - to_world(aim), axis=1).max()))
    return worst


def measure():
    P = np.vstack([r6.world_positions(scene, r["gid"]) for r in sheet_recs])
    return obs.observe(world.frame(), {
        "frac_below_nip": float(np.mean(P[:, 1] < -0.02)),
        "frac_on_floor": float(np.mean(P[:, 1] < GROUND_Y + 0.15)),
        "roller_track_err": roller_track_err(),
    })


base = measure()
capture.write_frame(0, force=True)
for i in range(args.frames):
    ms = runner.step(on_frame=measure)
    capture.write_frame(i + 1)
    if i % 10 == 0 or i == args.frames - 1:
        t = obs.rows[-1]
        r6.print_frame(i, ms, runner.frame_stats[-1],
                       f"through={t['frac_below_nip']:.2f} floor={t['frac_on_floor']:.2f} "
                       f"ymin={t['y_min']:+.3f} ymax={t['y_max']:+.3f} track={t['roller_track_err']*1e3:.1f}mm "
                       f"disp={t['mean_disp']*1e3:.2f}mm")
final_sanity = r6.run_sanity_check(world, "final") if not args.no_sanity else {"skipped": True}

rows = obs.rows
last = rows[-1]
extra_obs = {
    "sheet_n_tris": N_TRIS,
    "roller_n_tets": N_TETS,
    "frac_below_nip_start": float(rows[0]["frac_below_nip"]),
    "frac_below_nip_final": float(last["frac_below_nip"]),
    "frac_on_floor_final": float(last["frac_on_floor"]),
    "roller_track_err_m_max": float(max(r["roller_track_err"] for r in rows)),
    "cloth_y_max_final": float(last["y_max"]),
    "passes_through": bool(last["frac_below_nip"] > 0.5),
}
extra = {"config": {"dt": DT, "n_sheets": N_SHEETS, "spacing": SPACING, "gap0": GAP0, "gap1": GAP1,
                    "omega": OMEGA, "roller": [ROLL_R, ROLL_L, ROLL_E]}}
r6.finish(runner, capture, obs, args, extra_obs, extra, init_sanity, final_sanity,
          EDGE_LEN, CLOTH_R, D_HAT)
