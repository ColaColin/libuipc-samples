"""Example 98 -- "Garments on a spinning rack".

Four procedural garments from the tumbler benchmark (a towel, a pillowcase,
a pair of shorts and a washcloth, built by `tumbler_geometry`) hang folded
over the four *tangential* bars of a square rotary-clothesline frame (a
"Hills Hoist": hub, four diagonal arms, four side bars, one welded affine
body).  The rotor sits on a fixed post through a vertical *revolute joint*
and is turned by an `AffineBodyDrivingRevoluteJoint` angle motor -- the
mixer scene's idiom (example 100) -- whose commanded angular velocity ramps
from rest to OMEGA_MAX and holds (~1.3 g at the bars).

Centrifugal behaviour is the point: because the bars are tangential, the
hanging halves of every garment swing *outward*, perpendicular to the bar,
flap, and the two-layer garments balloon; friction on the bar is all that
keeps them from sliding over it and flying off.

Observables: the rotor tracks its commanded angle; the garments' hems move
outward (hem radius grows with the spin); no garment leaves the rack (all
cloth stays well above the floor and inside the domain); the cloth never
settles while the rack turns.

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
from uipc.geometry import ground
from uipc.constitution import (AffineBodyConstitution, AffineBodyRevoluteJoint,
                               AffineBodyDrivingRevoluteJoint)

SCENE_NAME = "98_rack"
DT = r6.DT
POST_H = 1.30
HUB_Y = 1.42                 # hub centre
BAR_HALF = 0.02              # square bar half-size
FRAME_HALF = 0.80            # square frame half-size (bar distance from the axis)
GARMENT_R = FRAME_HALF       # where the garments hang (distance from the axis)
OMEGA_MAX = 4.0              # rad/s (~0.64 rev/s; 1.3 g at the bars)
RAMP_S = 2.0
DOMAIN_R = 2.2

ap = r6.build_argparser(SCENE_NAME, 300)
r6.add_r6_args(ap, 0.012)
args = ap.parse_args()
EDGE_LEN = args.edge_len

r6.configure_benchmark_timers()
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Error")))
workspace = args.workspace or AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

# --------------------------------------------------------------------------
# garments first (contact resolution), then fold them over the arms
GARMENTS = [
    ("towel",      lambda h, gap: G.build_sheet(1.00, 0.70, h),                    0.0),
    ("pillowcase", lambda h, gap: G.build_bag(0.80, 0.60, h, gap=gap),            90.0),
    ("shorts",     lambda h, gap: G.build_trousers(0.50, 0.25, 0.55, 0.08, h, gap=gap), 180.0),
    ("washcloth",  lambda h, gap: G.build_sheet(0.60, 0.50, h),                  270.0),
]
probe = [b(EDGE_LEN, 0.01) for _n, b, _a in GARMENTS]
h_min = min(G.mesh_stats(V, F)["min_tri_height"] for V, F in probe)
CLOTH_R, D_HAT = r6.contact_resolution(h_min)
GAP = r6.layer_gap(CLOTH_R, D_HAT)
R_FOLD = BAR_HALF * math.sqrt(2.0) + GAP + 0.006

built = []
for (name, builder, ang_deg), _p in zip(GARMENTS, probe):
    V, F = builder(EDGE_LEN, GAP)
    a = math.radians(ang_deg)
    out_dir = np.array([math.cos(a), 0.0, math.sin(a)])       # side of the frame
    bar_dir = np.array([-math.sin(a), 0.0, math.cos(a)])      # tangential bar
    center = out_dir * GARMENT_R + np.array([0.0, HUB_Y, 0.0])
    Vw = r6.fold_over_bar(V, R_FOLD, center, bar_dir, (0.0, 1.0, 0.0))
    built.append(dict(name=name, V=Vw, V_rest=V, F=F, angle=ang_deg))

config = r6.common_config(D_HAT, tol_rate=args.tol_rate or 1e-4)
scene = Scene(config)
ct = scene.contact_tabular()
ct.default_model(0.4, r6.CONTACT_RESISTANCE)
elem_rack = ct.create("rack")
elem_cloth = ct.create("cloth")
ct.insert(elem_rack, elem_rack, 0.0, r6.CONTACT_RESISTANCE, False)   # post/rotor: one machine
ct.insert(elem_rack, elem_cloth, 0.6, r6.CONTACT_RESISTANCE)
ct.insert(elem_cloth, elem_cloth, 0.3, r6.CONTACT_RESISTANCE)

abd = AffineBodyConstitution()

# --- post (fixed) --------------------------------------------------------------
pV, pF = r6.box_tri((0.12, POST_H, 0.12), (0.0, POST_H / 2 + 0.01, 0.0))
post_mesh = r6.make_abd_trimesh(pV, pF, abd, 1.0e8, 500.0, elem_rack, fixed=True)
post_obj = scene.objects().create("post")
post_slot, _ = post_obj.geometries().create(post_mesh)

# --- rotor: hub + four diagonal arms + four tangential side bars, welded -----
def rot_y(V, deg):
    a = math.radians(deg)
    c, s_ = math.cos(a), math.sin(a)
    R = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
    return V @ R.T


DIAG = FRAME_HALF * math.sqrt(2.0)
parts = [r6.box_tri((0.22, 0.10, 0.22), (0.0, HUB_Y, 0.0))]
for deg in (45.0, 135.0):
    aV, aF = r6.box_tri((2 * DIAG + 2 * BAR_HALF, 2 * BAR_HALF, 2 * BAR_HALF), (0.0, HUB_Y, 0.0))
    parts.append((rot_y(aV, deg), aF))
parts.append(r6.box_tri((2 * FRAME_HALF + 2 * BAR_HALF, 2 * BAR_HALF, 2 * BAR_HALF), (0.0, HUB_Y, FRAME_HALF)))
parts.append(r6.box_tri((2 * FRAME_HALF + 2 * BAR_HALF, 2 * BAR_HALF, 2 * BAR_HALF), (0.0, HUB_Y, -FRAME_HALF)))
parts.append(r6.box_tri((2 * BAR_HALF, 2 * BAR_HALF, 2 * FRAME_HALF + 2 * BAR_HALF), (FRAME_HALF, HUB_Y, 0.0)))
parts.append(r6.box_tri((2 * BAR_HALF, 2 * BAR_HALF, 2 * FRAME_HALF + 2 * BAR_HALF), (-FRAME_HALF, HUB_Y, 0.0)))
rV, rF = r6.weld(parts)
rotor_mesh = r6.make_abd_trimesh(rV, rF, abd, 1.0e8, 500.0, elem_rack)
rotor_obj = scene.objects().create("rotor")
rotor_slot, _ = rotor_obj.geometries().create(rotor_mesh)

# --- driven revolute joint about the vertical axis --------------------------------
revolute = AffineBodyRevoluteJoint()
joint_mesh = revolute.create_geometry(
    np.array([[0.0, HUB_Y - 0.5, 0.0]]), np.array([[0.0, HUB_Y + 0.5, 0.0]]),
    [post_slot], np.array([0], dtype=np.int32),
    [rotor_slot], np.array([0], dtype=np.int32),
    np.array([200.0], dtype=np.float64))
driving = AffineBodyDrivingRevoluteJoint()
driving.apply_to(joint_mesh, np.array([200.0], dtype=np.float64))
joint_obj = scene.objects().create("hinge")
joint_obj.geometries().create(joint_mesh)


def omega_at(t):
    return OMEGA_MAX * min(1.0, max(0.0, t / RAMP_S))


def commanded_angle(frame):
    t = frame * DT
    if t <= RAMP_S:
        return 0.5 * OMEGA_MAX / RAMP_S * t * t
    return 0.5 * OMEGA_MAX * RAMP_S + OMEGA_MAX * (t - RAMP_S)


def animate_drive(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    angles = view(geo.edges().find("angle"))
    view(geo.edges().find("driving/is_constrained"))[:] = 1
    aim = view(geo.edges().find("aim_angle"))
    t = info.frame() * info.dt()
    aim[0] = angles[0] + info.dt() * omega_at(t)


scene.animator().insert(joint_obj, animate_drive)

# --- garments -------------------------------------------------------------------------
cloth = r6.ClothMaterial(CLOTH_R, elem_cloth)
for g in built:
    m = cloth.make(g["V"], g["F"])
    o = scene.objects().create(g["name"])
    g["slot"], _ = o.geometries().create(m)
    g["gid"] = o.geometries().ids()[0]

ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(0.0))

# --------------------------------------------------------------------------
world.init(scene)
init_sanity = r6.run_sanity_check(world, "post-init") if not args.no_sanity else {"skipped": True}
print("SANITY(init):", init_sanity.get("result"), "clean=", init_sanity.get("clean"), flush=True)
if not world.is_valid():
    raise SystemExit("world invalid after init -- sanity check failed")

N_TRIS = int(sum(len(g["F"]) for g in built))
print(f"{SCENE_NAME}: {len(built)} garments, {N_TRIS} tris, edge_len={EDGE_LEN*1e3:.1f}mm "
      f"h_min={h_min*1e3:.2f}mm r={CLOTH_R*1e3:.2f}mm d_hat={D_HAT*1e3:.2f}mm "
      f"omega_max={OMEGA_MAX} rad/s frames={args.frames}", flush=True)

capture = r6.Capture(args.capture, scene, up_dir="y_up", ground_height=0.0,
                     stride=args.capture_stride, obj_dir=args.obj)
capture.bind([(g["name"], g["slot"]) for g in built] + [("rotor", rotor_slot), ("post", post_slot)])
runner = r6.Runner(world, engine, scene, args, SCENE_NAME)
obs = r6.ClothObserver(scene, [(g["name"], g["gid"], g["V"], g["F"]) for g in built],
                       DT, CLOTH_R, D_HAT, bounds={"radius_max": DOMAIN_R}, ground_y=0.0)


def rotor_angle():
    A = np.asarray(rotor_slot.geometry().transforms().view()[0]).reshape(4, 4)
    return math.atan2(-A[2, 0], A[0, 0])


HEM = {g["name"]: np.flatnonzero(g["V"][:, 1] < np.quantile(g["V"][:, 1], 0.15)) for g in built}


def measure():
    ex = {"rotor_angle": rotor_angle()}
    for g in built:
        P = r6.world_positions(scene, g["gid"])
        r = np.hypot(P[:, 0], P[:, 2])
        ex[f"r_{g['name']}"] = float(r.mean())
        ex[f"rhem_{g['name']}"] = float(r[HEM[g["name"]]].mean())
        ex[f"ymin_{g['name']}"] = float(P[:, 1].min())
        ex[f"ymax_{g['name']}"] = float(P[:, 1].max())
    return obs.observe(world.frame(), ex)


base = measure()
capture.write_frame(0, force=True)
for i in range(args.frames):
    ms = runner.step(on_frame=measure)
    capture.write_frame(i + 1)
    if i % 10 == 0 or i == args.frames - 1:
        t = obs.rows[-1]
        r6.print_frame(i, ms, runner.frame_stats[-1],
                       f"rotor={math.degrees(t['rotor_angle']):+8.1f}deg ymin={t['y_min']:+.3f} "
                       f"rhem_towel={t['rhem_towel']:.3f} disp={t['mean_disp']*1e3:.2f}mm")
final_sanity = r6.run_sanity_check(world, "final") if not args.no_sanity else {"skipped": True}

rows = obs.rows
ang = np.unwrap([r["rotor_angle"] for r in rows])
cmd = np.array([commanded_angle(r["frame"]) for r in rows])
err = np.degrees(ang - ang[0] - cmd)
last = rows[-1]
lift = {g["name"]: float(last[f"rhem_{g['name']}"] - base[f"rhem_{g['name']}"]) for g in built}
swing_max = {g["name"]: float(max(r[f"rhem_{g['name']}"] for r in rows) - base[f"rhem_{g['name']}"]) for g in built}
extra_obs = {
    "garment_n_tris": N_TRIS,
    "rotor_track_err_deg_max": float(np.max(np.abs(err))),
    "rotor_turn_deg": float(math.degrees(ang[-1] - ang[0])),
    "rotor_commanded_deg": float(math.degrees(cmd[-1])),
    "hem_radius_swing_final_m": lift,
    "hem_radius_swing_max_m": swing_max,
    "garment_y_min_final": {g["name"]: float(last[f"ymin_{g['name']}"]) for g in built},
    "garment_y_max_over_run": {g["name"]: float(max(r[f"ymax_{g['name']}"] for r in rows)) for g in built},
    # still hanging: nothing on the floor, and no garment ever rises more than
    # a fold radius above the bar (which is what flying off over it looks like)
    "garments_on_rack": bool(min(last[f"ymin_{g['name']}"] for g in built) > 0.5
                             and max(max(r[f"ymax_{g['name']}"] for r in rows) for g in built)
                             < HUB_Y + R_FOLD + 0.12),
    "centrifugal_swing_ok": bool(min(swing_max.values()) > 0.05),
}
extra = {"config": {"dt": DT, "omega_max": OMEGA_MAX, "ramp_s": RAMP_S, "garment_r": GARMENT_R}}
r6.finish(runner, capture, obs, args, extra_obs, extra, init_sanity, final_sanity,
          EDGE_LEN, CLOTH_R, D_HAT)
