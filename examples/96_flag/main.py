"""Example 96 -- "Flag in a gust".

A single large flag (Baraff-Witkin strain-limiting shell + discrete shell
bending, the tumbler's calibrated cloth) is pinned along its hoist to a fixed
affine-body pole and driven by an animated wind field: a per-vertex
aerodynamic form-drag force (FiniteElementExternalForce, updated every frame
by the scene animator from the cloth's *current* normals and velocities).

The wind never holds still -- the mean speed carries two incommensurate
gust harmonics, the direction yaws, and a travelling spatial gust runs along
the fly -- so the flag has no steady state to relax into.  The observable is
that it keeps fluttering: mean per-frame vertex displacement in the last
quarter of the run stays well above the "settled" threshold while the fabric
stays finite, un-inverted and inside the domain around the pole.

Wind model (per vertex i, current normal n_i, current velocity v_i, lumped
area A_i):  v_rel = v_wind(x_i, t) - v_i,
            F_i = rho_air * A_i * ( C_n (v_rel . n_i) |v_rel . n_i| n_i
                                  + C_t (v_rel - (v_rel . n_i) n_i) )
with C_n = 1.2 (flat-plate form drag) and C_t = 0.05 (skin friction).

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
from uipc.constitution import AffineBodyConstitution, FiniteElementExternalForce

SCENE_NAME = "96_flag"
DT = r6.DT

FLAG_L = 1.6                 # fly (x), metres
FLAG_H = 1.0                 # hoist (y)
POLE_R = 0.035
POLE_H = 2.1
FLAG_TOP = 1.95
RHO_AIR = 1.2
C_N, C_T = 1.2, 0.05
U0 = 7.0                     # mean wind speed (m/s), gusting 0.35 .. 1.75 x
DOMAIN_R = 3.0               # sanity: no vertex further than this from the pole axis

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
# cloth first: it sets the contact resolution
V0, F0 = G.build_sheet(FLAG_L, FLAG_H, EDGE_LEN)      # flat in x-y, centred
st = G.mesh_stats(V0, F0)
CLOTH_R, D_HAT = r6.contact_resolution(st["min_tri_height"])
gap = POLE_R + r6.layer_gap(CLOTH_R, D_HAT)            # hoist stands off the pole
Vf = V0.copy()
Vf[:, 0] += FLAG_L / 2 + gap                            # hoist at x = gap
Vf[:, 1] += FLAG_TOP - FLAG_H / 2                       # top edge at FLAG_TOP
Vf[:, 2] += 0.0

config = r6.common_config(D_HAT, tol_rate=args.tol_rate or 1e-4)
scene = Scene(config)

ct = scene.contact_tabular()
ct.default_model(0.3, r6.CONTACT_RESISTANCE)
elem_pole = ct.create("pole")
elem_cloth = ct.create("cloth")
ct.insert(elem_pole, elem_cloth, 0.4, r6.CONTACT_RESISTANCE)
ct.insert(elem_cloth, elem_cloth, 0.3, r6.CONTACT_RESISTANCE)

# --- pole: fixed affine body -------------------------------------------------
abd = AffineBodyConstitution()
pV, pF = r6.cylinder_tri(POLE_R, POLE_H, axis="y", center=(0.0, POLE_H / 2 + 0.01, 0.0), n_seg=28, n_len=6)
finV, finF = r6.cylinder_tri(POLE_R * 2.2, 0.06, axis="y", center=(0.0, POLE_H + 0.055, 0.0), n_seg=28, n_len=1)
pole_mesh = r6.make_abd_trimesh(pV, pF, abd, 1.0e8, 500.0, elem_pole, fixed=True)
pole_obj = scene.objects().create("pole")
pole_slot, _ = pole_obj.geometries().create(pole_mesh)
fin_mesh = r6.make_abd_trimesh(finV, finF, abd, 1.0e8, 500.0, elem_pole, fixed=True)
fin_obj = scene.objects().create("finial")
fin_slot, _ = fin_obj.geometries().create(fin_mesh)

# --- flag ---------------------------------------------------------------------
cloth = r6.ClothMaterial(CLOTH_R, elem_cloth)
flag_mesh = cloth.make(Vf, F0)
hoist = np.flatnonzero(Vf[:, 0] < gap + 1e-6)
is_fixed = flag_mesh.vertices().find(builtin.is_fixed)
if is_fixed is None:
    is_fixed = flag_mesh.vertices().create(builtin.is_fixed, 0)
fv = view(is_fixed)
fv[hoist] = 1
ext = FiniteElementExternalForce()
ext.apply_to(flag_mesh, np.zeros(3))
flag_obj = scene.objects().create("flag")
flag_slot, _ = flag_obj.geometries().create(flag_mesh)

# lumped vertex areas (rest), for the drag force
_a0 = r6.ClothObserver._areas(Vf, F0)
AREA = np.zeros(len(Vf))
for k in range(3):
    np.add.at(AREA, F0[:, k], _a0 / 3.0)
FREE = np.ones(len(Vf), dtype=bool)
FREE[hoist] = False

wind_state = {"prev": Vf.copy(), "u_mean": [], "force_max": []}


def wind_velocity(P, t):
    """Gusty wind, mostly along +x (the fly direction), yawing in z."""
    U = U0 * (1.0 + 0.45 * math.sin(2 * math.pi * 0.55 * t)
              + 0.30 * math.sin(2 * math.pi * 1.85 * t + 1.0))
    yaw = math.radians(18.0) * math.sin(2 * math.pi * 0.37 * t + 0.4)
    d = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
    W = np.repeat(U * d[None, :], len(P), axis=0)
    # travelling spatial gust along the fly + a vertical component
    W[:, 2] += 0.35 * U0 * np.sin(4.0 * P[:, 0] - 9.0 * t)
    W[:, 1] += 0.20 * U0 * np.sin(3.0 * P[:, 1] + 6.0 * t + 2.0)
    return W


def animate_wind(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    P = np.asarray(geo.positions().view()).reshape(-1, 3)
    t = info.frame() * info.dt()
    vel = (P - wind_state["prev"]) / info.dt() if info.frame() > 1 else np.zeros_like(P)
    wind_state["prev"] = P.copy()
    a, b, c = P[F0[:, 0]], P[F0[:, 1]], P[F0[:, 2]]
    fn = np.cross(b - a, c - a)
    N = np.stack([np.bincount(F0.ravel(), weights=np.repeat(fn[:, d], 3), minlength=len(P))
                  for d in range(3)], axis=1)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    vrel = wind_velocity(P, t) - vel
    vn = np.einsum("ij,ij->i", vrel, N)
    Fv = RHO_AIR * AREA[:, None] * (C_N * (vn * np.abs(vn))[:, None] * N
                                    + C_T * (vrel - vn[:, None] * N))
    Fv[~FREE] = 0.0
    fview = view(geo.vertices().find("external_force"))
    fview[:] = Fv.reshape(-1, 3, 1)
    ic = view(geo.vertices().find(builtin.is_constrained))
    ic[:] = FREE.astype(np.int32)
    wind_state["u_mean"].append(float(np.linalg.norm(vrel.mean(axis=0))))
    wind_state["force_max"].append(float(np.linalg.norm(Fv, axis=1).max()))


scene.animator().insert(flag_obj, animate_wind)

ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(0.0))

# --------------------------------------------------------------------------
world.init(scene)
init_sanity = r6.run_sanity_check(world, "post-init") if not args.no_sanity else {"skipped": True}
print("SANITY(init):", init_sanity.get("result"), "clean=", init_sanity.get("clean"), flush=True)
if not world.is_valid():
    raise SystemExit("world invalid after init -- sanity check failed")

flag_gid = flag_obj.geometries().ids()[0]
print(f"{SCENE_NAME}: flag {len(Vf)} verts / {len(F0)} tris, edge_len={EDGE_LEN*1e3:.1f}mm "
      f"h_min={st['min_tri_height']*1e3:.2f}mm r={CLOTH_R*1e3:.2f}mm d_hat={D_HAT*1e3:.2f}mm "
      f"pinned={len(hoist)} frames={args.frames}", flush=True)

capture = r6.Capture(args.capture, scene, up_dir="y_up", ground_height=0.0,
                     stride=args.capture_stride, obj_dir=args.obj)
capture.bind([("flag", flag_slot), ("pole", pole_slot), ("finial", fin_slot)])
runner = r6.Runner(world, engine, scene, args, SCENE_NAME)
obs = r6.ClothObserver(scene, [("flag", flag_gid, Vf, F0)], DT, CLOTH_R, D_HAT,
                       bounds={"radius_max": DOMAIN_R, "y_max": POLE_H + 0.5}, ground_y=0.0)


def measure():
    P = r6.world_positions(scene, flag_gid)
    tip = P[np.argmax(Vf[:, 0])]
    return obs.observe(world.frame(), {
        "fly_tip": tip.tolist(),
        "fly_extent_x": float(P[:, 0].max()),
        "wind_u": wind_state["u_mean"][-1] if wind_state["u_mean"] else 0.0,
        "force_max": wind_state["force_max"][-1] if wind_state["force_max"] else 0.0,
    })


obs.observe(0)
capture.write_frame(0, force=True)
for i in range(args.frames):
    ms = runner.step(on_frame=measure)
    capture.write_frame(i + 1)
    if i % 10 == 0 or i == args.frames - 1:
        t = obs.rows[-1]
        r6.print_frame(i, ms, runner.frame_stats[-1],
                       f"tip=({t['fly_tip'][0]:+.2f},{t['fly_tip'][1]:+.2f},{t['fly_tip'][2]:+.2f}) "
                       f"disp={t['mean_disp']*1e3:.2f}mm vmax={t['max_speed']:.1f} U={t['wind_u']:.1f}")
final_sanity = r6.run_sanity_check(world, "final") if not args.no_sanity else {"skipped": True}

rows = obs.rows[1:]
tips = np.array([r["fly_tip"] for r in rows])
q = max(1, len(rows) // 4)
tip_sd_last = float(np.std(tips[-q:], axis=0).sum()) if len(rows) else 0.0
extra_obs = {
    "flag_n_tris": int(len(F0)),
    "fly_tip_std_last_quarter_m": tip_sd_last,
    "fly_extent_x_mean": float(np.mean([r["fly_extent_x"] for r in rows])),
    "flutter_ok": bool(tip_sd_last > 0.02),
}
extra = {"config": {"dt": DT, "U0": U0, "flag": [FLAG_L, FLAG_H], "pole_r": POLE_R}}
payload = r6.finish(runner, capture, obs, args, extra_obs, extra, init_sanity, final_sanity,
                    EDGE_LEN, CLOTH_R, D_HAT)
