"""Example 97 -- "Drape over tumbling rigids" (a take-up roll).

A large sheet of the tumbler's calibrated cloth hangs in the air with its
leading edge *stitched* (`SoftVertexStitch`, cloth vertex <-> affine-body
vertex) to a kinematic take-up axle -- an affine body driven exactly like the
tumbler drum (SoftTransformConstraint fed an absolute aim pose R_x(omega t)
every frame).  Three free affine bodies (a ball, a chain link and a rigid
bunny) rest on the floor under the sheet.  The sheet falls and drapes over
them; the axle then winds it in, layer over layer, dragging the sheet across
the floor with the bodies rolling and tumbling under it until they are
pulled up against -- or into -- the roll.

Why a stitched take-up rather than a free sheet on a spinning cross: a free
sheet slides off a rotating spit within one turn and settles into a heap
(tested, both at mu = 0.5 and at mu = 0.8 / 12 rpm); the stitched edge keeps
the cloth engaged for the whole run, so the wrapping and the cloth-rigid
friction are sustained rather than transient.

Observables: the wound fraction of the sheet (vertices within the roll
radius of the axle) grows monotonically; every free body is displaced by the
cloth; the stitches hold (pair distance stays small); the axle tracks its
commanded angle; the sheet never stops moving.  The domain is a ground plane
plus four half-plane walls.

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
from uipc.geometry import (SimplicialComplexIO, ground, halfplane, label_surface,
                           label_triangle_orient, flip_inward_triangles)
from uipc.constitution import (AffineBodyConstitution, SoftTransformConstraint,
                               SoftVertexStitch)

SCENE_NAME = "97_drape"
DT = r6.DT
TETMESH = AssetDir.tetmesh_path()
SHEET_L, SHEET_W = 2.4, 3.2            # x (along the axle), s (winding direction)
AXLE_R, AXLE_L = 0.10, 2.7
AXLE_Y, AXLE_Z = 0.75, -1.15
RPM_DEFAULT = 40.0
STITCH_KAPPA = 1.0e5
RIGID_DENSITY = 120.0                  # foam-light props: a 2 kg sheet has to be able to move them
BIN_X, BIN_Z = 2.0, 1.8

ap = r6.build_argparser(SCENE_NAME, 300)
r6.add_r6_args(ap, 0.018)
ap.add_argument("--rpm", type=float, default=RPM_DEFAULT)
ap.add_argument("--mu-axle", type=float, default=0.5, help="axle-cloth friction")
args = ap.parse_args()
EDGE_LEN = args.edge_len
RPM = args.rpm
OMEGA = 2 * math.pi * RPM / 60.0

r6.configure_benchmark_timers()
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Error")))
workspace = args.workspace or AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

# --------------------------------------------------------------------------
V0, F0 = G.build_sheet(SHEET_L, SHEET_W, EDGE_LEN)     # x = L, y = W (-> z)
st = G.mesh_stats(V0, F0)
CLOTH_R, D_HAT = r6.contact_resolution(st["min_tri_height"])
GAP = r6.layer_gap(CLOTH_R, D_HAT)
# Sheet layout along its winding coordinate s (0 = leading edge at the axle):
#   ramp   -- straight from the axle's +z face down to the floor
#   bottom -- flat on the floor (the props sit on this part)
#   fold   -- a half-circle up to Y_TOP
#   top    -- folded back over the props, hanging in the air until it falls
# so the top layer *drapes* onto the props and the bottom layer, once the axle
# pulls, carries them toward the roll instead of sliding out from under them.
Y0 = GAP                                   # bottom layer: one contact gap above the floor
Y_TOP = 0.56
Z_B, L2 = -0.45, 0.60
A = np.array([AXLE_Y, AXLE_Z + AXLE_R + GAP])       # (y, z) of the leading edge
B = np.array([Y0, Z_B])
L1 = float(np.linalg.norm(B - A))
RHO = (Y_TOP - Y0) / 2
L_ARC = math.pi * RHO
Z_C = Z_B + L2


def lay_sheet(sv):
    y = np.empty_like(sv)
    z = np.empty_like(sv)
    m = sv <= L1
    t = sv[m] / L1
    y[m], z[m] = A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t
    m = (sv > L1) & (sv <= L1 + L2)
    y[m], z[m] = Y0, Z_B + (sv[m] - L1)
    m = (sv > L1 + L2) & (sv <= L1 + L2 + L_ARC)
    phi = (sv[m] - L1 - L2) / RHO
    y[m], z[m] = Y0 + RHO - RHO * np.cos(phi), Z_C + RHO * np.sin(phi)
    m = sv > L1 + L2 + L_ARC
    y[m], z[m] = Y_TOP, Z_C - (sv[m] - L1 - L2 - L_ARC)
    return y, z


_s = V0[:, 1] + SHEET_W / 2
_y, _z = lay_sheet(_s)
Vc = np.stack([V0[:, 0], _y, _z], axis=1)
LEAD = np.flatnonzero(_s < 1e-6)

config = r6.common_config(D_HAT, tol_rate=args.tol_rate or 1e-4)
scene = Scene(config)
ct = scene.contact_tabular()
ct.default_model(0.4, r6.CONTACT_RESISTANCE)
elem_axle = ct.create("axle")
elem_rigid = ct.create("rigid")
elem_cloth = ct.create("cloth")
ct.insert(elem_axle, elem_cloth, args.mu_axle, r6.CONTACT_RESISTANCE)
ct.insert(elem_axle, elem_rigid, 0.3, r6.CONTACT_RESISTANCE)
ct.insert(elem_rigid, elem_cloth, 0.5, r6.CONTACT_RESISTANCE)
ct.insert(elem_rigid, elem_rigid, 0.3, r6.CONTACT_RESISTANCE)
ct.insert(elem_cloth, elem_cloth, 0.3, r6.CONTACT_RESISTANCE)

abd = AffineBodyConstitution()
stc = SoftTransformConstraint()

# --- take-up axle: kinematic, rings spaced ~one cloth edge so every leading-edge
#     vertex has an axle vertex to stitch to ------------------------------------
N_SEG = 24
N_LEN = int(min(240, round(AXLE_L / EDGE_LEN)))
aV, aF = r6.cylinder_tri(AXLE_R, AXLE_L, axis="x", n_seg=N_SEG, n_len=N_LEN)
AXLE_T0 = r6.transform4(translate=(0.0, AXLE_Y, AXLE_Z))
axle_mesh = r6.make_abd_trimesh(aV, aF, abd, 1.0e8, 500.0, elem_axle, transform=AXLE_T0)
stc.apply_to(axle_mesh, np.array([100.0, 100.0], dtype=np.float64))
axle_obj = scene.objects().create("axle")
axle_slot, _ = axle_obj.geometries().create(axle_mesh)
# ring-line vertices facing +z (theta = pi before the axis permutation)
J0 = N_SEG // 2
ring_ids = np.array([k * N_SEG + J0 for k in range(N_LEN + 1)])
assert np.allclose(aV[ring_ids, 2], AXLE_R) and np.allclose(aV[ring_ids, 1], 0.0)


def commanded(frame):
    # surface at the +z face moves down (-y): rotation about +x
    return AXLE_T0 @ r6.rot_axis((1, 0, 0), OMEGA * DT * frame)


def animate_axle(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    view(geo.instances().find(builtin.is_constrained))[0] = 1
    view(geo.instances().find(builtin.aim_transform))[0] = commanded(info.frame())


scene.animator().insert(axle_obj, animate_axle)

# --- free rigid bodies ---------------------------------------------------------
def read_tet(path, scale):
    io = SimplicialComplexIO(r6.transform4(scale=scale))
    m = io.read(path)
    label_surface(m)
    label_triangle_orient(m)
    return flip_inward_triangles(m)


free = []
# (name, asset, scale, floor position, yaw about y, pitch about x)
FREE_BODIES = [
    # on the bottom layer, under the folded-back top layer
    ("ball", f"{TETMESH}/ball.msh", 0.07, (0.60, 0.0, -0.05), 0.0, 0.0),
    ("link", f"{TETMESH}/link.msh", 0.40, (-0.80, 0.0, -0.15), 0.7, math.pi / 2),   # lying flat
    ("bunny", f"{TETMESH}/bunny0.msh", 0.30, (-0.20, 0.0, -0.05), math.pi / 2, 0.0),
]
for name, path, scale, pos, yaw, pitch in FREE_BODIES:
    m = read_tet(path, scale)
    P = np.asarray(m.positions().view()).reshape(-1, 3)
    R = r6.transform4(rotate=((0, 1, 0), yaw)) @ r6.transform4(rotate=((1, 0, 0), pitch))
    Pr = P @ R[:3, :3].T
    lift = Y0 + GAP - Pr[:, 1].min()          # resting on the bottom layer
    T = r6.transform4(translate=(pos[0], lift, pos[2])) @ R
    abd.apply_to(m, 5.0e7, RIGID_DENSITY)
    elem_rigid.apply_to(m)
    m.instances().resize(1)
    r6.set_instance_transform(m, 0, T)
    view(m.instances().find(builtin.is_fixed))[0] = 0
    o = scene.objects().create(name)
    slot, _ = o.geometries().create(m)
    free.append((name, slot, o.geometries().ids()[0]))

# --- cloth + stitches ------------------------------------------------------------
cloth = r6.ClothMaterial(CLOTH_R, elem_cloth)
sheet_mesh = cloth.make(Vc, F0)
sheet_obj = scene.objects().create("sheet")
sheet_slot, _ = sheet_obj.geometries().create(sheet_mesh)

ring_x = aV[ring_ids, 0]
pairs = np.array([[int(ring_ids[np.argmin(np.abs(ring_x - Vc[v, 0]))]), int(v)] for v in LEAD],
                 dtype=np.int32)
svs = SoftVertexStitch()
stitch_geo = svs.create_geometry((axle_slot, sheet_slot), pairs, STITCH_KAPPA, GAP)
stitch_obj = scene.objects().create("stitch")
stitch_obj.geometries().create(stitch_geo)

# --- domain ------------------------------------------------------------------------
bin_obj = scene.objects().create("bin")
bin_obj.geometries().create(ground(0.0))
for P, N in (((BIN_X, 0, 0), (-1, 0, 0)), ((-BIN_X, 0, 0), (1, 0, 0)),
             ((0, 0, BIN_Z), (0, 0, -1)), ((0, 0, -BIN_Z), (0, 0, 1))):
    bin_obj.geometries().create(halfplane(np.array(P, dtype=float), np.array(N, dtype=float)))

# --------------------------------------------------------------------------
world.init(scene)
init_sanity = r6.run_sanity_check(world, "post-init") if not args.no_sanity else {"skipped": True}
print("SANITY(init):", init_sanity.get("result"), "clean=", init_sanity.get("clean"), flush=True)
if not world.is_valid():
    raise SystemExit("world invalid after init -- sanity check failed")

sheet_gid = sheet_obj.geometries().ids()[0]
axle_gid = axle_obj.geometries().ids()[0]
print(f"{SCENE_NAME}: sheet {len(Vc)} verts / {len(F0)} tris (+axle {len(aF)}), {len(pairs)} stitches, "
      f"edge_len={EDGE_LEN*1e3:.1f}mm h_min={st['min_tri_height']*1e3:.2f}mm r={CLOTH_R*1e3:.2f}mm "
      f"d_hat={D_HAT*1e3:.2f}mm {RPM:g} rpm frames={args.frames}", flush=True)

capture = r6.Capture(args.capture, scene, up_dir="y_up", ground_height=0.0,
                     stride=args.capture_stride, obj_dir=args.obj)
capture.bind([("sheet", sheet_slot), ("axle", axle_slot)] + [(n, s) for n, s, _ in free])
runner = r6.Runner(world, engine, scene, args, SCENE_NAME)
obs = r6.ClothObserver(scene, [("sheet", sheet_gid, Vc, F0)], DT, CLOTH_R, D_HAT,
                       bounds={"abs_x_max": BIN_X, "abs_z_max": BIN_Z}, ground_y=0.0)


def axle_angle():
    A = np.asarray(axle_slot.geometry().transforms().view()[0]).reshape(4, 4)
    return math.atan2(A[2, 1], A[1, 1])


def measure():
    P = r6.world_positions(scene, sheet_gid)
    A = r6.world_positions(scene, axle_gid)
    rad = np.hypot(P[:, 1] - AXLE_Y, P[:, 2] - AXLE_Z)
    d = np.linalg.norm(P[pairs[:, 1]] - A[pairs[:, 0]], axis=1)
    ex = {"wound_frac": float(np.mean(rad < AXLE_R + 0.15)),
          "stitch_dist_max": float(d.max()),
          "axle_angle": axle_angle()}
    for name, _slot, gid in free:
        ex[f"c_{name}"] = r6.world_positions(scene, gid).mean(axis=0).tolist()
    return obs.observe(world.frame(), ex)


base = measure()
capture.write_frame(0, force=True)
for i in range(args.frames):
    ms = runner.step(on_frame=measure)
    capture.write_frame(i + 1)
    if i % 10 == 0 or i == args.frames - 1:
        t = obs.rows[-1]
        r6.print_frame(i, ms, runner.frame_stats[-1],
                       f"ymin={t['y_min']:+.3f} wound={t['wound_frac']:.2f} stitch={t['stitch_dist_max']*1e3:.1f}mm "
                       f"axle={math.degrees(t['axle_angle']):+7.1f}deg disp={t['mean_disp']*1e3:.2f}mm")
final_sanity = r6.run_sanity_check(world, "final") if not args.no_sanity else {"skipped": True}

rows = obs.rows
ang = np.unwrap([r["axle_angle"] for r in rows])
cmd = OMEGA * DT * np.array([r["frame"] for r in rows])
err = np.degrees(ang - ang[0] - cmd)
last = rows[-1]
moved = {name: float(np.linalg.norm(np.array(last[f"c_{name}"]) - np.array(base[f"c_{name}"])))
         for name, _s, _g in free}
wound = np.array([r["wound_frac"] for r in rows])
extra_obs = {
    "sheet_n_tris": int(len(F0)),
    "n_stitches": int(len(pairs)),
    "wound_frac_start": float(wound[0]),
    "wound_frac_max": float(wound.max()),
    "wound_frac_final": float(wound[-1]),
    "stitch_dist_max_m": float(max(r["stitch_dist_max"] for r in rows)),
    "axle_track_err_deg_max": float(np.max(np.abs(err))),
    "axle_turn_deg": float(math.degrees(ang[-1] - ang[0])),
    "free_body_displacement_m": moved,
    "wrap_ok": bool(wound[-1] > 0.2 and wound[-1] >= 0.9 * wound.max()),
    "stitch_ok": bool(max(r["stitch_dist_max"] for r in rows) < 0.05),
    "bodies_dragged": bool(sorted(moved.values())[-2] > 0.15),   # at least two bodies moved
}
extra = {"config": {"dt": DT, "rpm": RPM, "sheet": [SHEET_L, SHEET_W], "axle": [AXLE_R, AXLE_L],
                    "stitch_kappa": STITCH_KAPPA, "bin": [BIN_X, BIN_Z]}}
r6.finish(runner, capture, obs, args, extra_obs, extra, init_sanity, final_sanity,
          EDGE_LEN, CLOTH_R, D_HAT)
