"""Example 95 -- Tumbler with garments: a rotating drum full of cloth.

A horizontal-axis tumbler (washing-machine drum) driven at a constant angular
velocity, with four thin-shell garments inside.  The workload is *dense
multi-layer cloth self-contact under continuous frictional sliding*: the
garments are lifted by three radial lifters, dropped, folded over each other
and dragged along the bore wall for the whole run.  Nothing settles.

World frame: +y up (gravity -y), the drum axis is +z.

Geometry is procedural (numpy only, `tumbler_geometry.py`) -- no binary assets.
The builders are a minimal port of the `cloth-machine` / `cloth-dataset`
garment and drum generators; see that module's header for the provenance.

Drum drive: a `SoftTransformConstraint` fed an **absolute** aim transform
R_z(omega * t) every frame, not a `RotatingMotor`.  `RotatingMotor` prescribes
only the per-frame rotation *increment* relative to the body's current pose
(so any tracking lag becomes permanent drift, with no absolute reference to
measure against) and it sets the translational strength to zero (so a drum
loaded with garments is free to fall out of place).  The absolute aim pins both
the axis position and the commanded angle, which makes "the drum tracks its
commanded angle" a measurable statement -- see `--verify`.

Usage:
  python main.py                     # GUI with run/stop
  python main.py --headless [N]      # N frames (default 180), benchmark output
  python main.py --headless --verify # same run plus the physical-soundness audit
  python main.py --headless 60 --edge-len=0.010   # fidelity sweep knob
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
from uipc import Animation, Logger, builtin, view
from uipc.core import Engine, World, Scene
from uipc.geometry import halfplane, label_surface, trimesh
from uipc.constitution import (AffineBodyConstitution, DiscreteShellBending,
                               ElasticModuli2D, SoftTransformConstraint,
                               StrainLimitingBaraffWitkinShell)

sys.path.append(str(Path(__file__).resolve().parent))
import tumbler_geometry as G

sys.path.append(str(Path(__file__).resolve().parents[1]))
from benchmark_utils import (configure_benchmark_timers, emit_benchmark_result,
                             report_timers_if_enabled, snapshot_frame_stats)


# --------------------------------------------------------------------------
# CLI: `--headless [N]` exactly as the sibling benchmark scenes; every other
# knob is `--key=value` so it never lands in the positional frame count.
# --------------------------------------------------------------------------
def _opt(name, default, cast=float):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return cast(a.split("=", 1)[1])
    return default


HEADLESS = "--headless" in sys.argv
VERIFY = "--verify" in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
N_FRAMES = int(_positional[0]) if _positional else 180

# --- validation-pass knobs (round 6 / V1).  All three default to a value that
# makes the code below take *exactly* the branch it took before they existed,
# so the default scene -- the round's instrument -- stays byte-stable and the
# manifest's comparability note still holds. -----------------------------------
#
#   --perturb-yaw=<rad>     add a rotation to ONE garment's initial yaw
#   --perturb-vertex=<m>    add an offset to ONE vertex's initial x coordinate
#   --perturb-garment=<i>   which garment (0=towel, 1=pillowcase, 2=shorts,
#                           3=washcloth); only read when a perturbation is set
#
# These exist to build the *perturbed-exact control arm*: a physically
# meaningless change of the initial state that breaks the bitwise path, so that
# "how far do two physically equivalent runs of this system drift apart on
# their own" becomes measurable and can be compared against the drift caused by
# a change of the solver's search direction.
PERTURB_YAW = _opt("perturb-yaw", 0.0)          # radians
PERTURB_VERTEX = _opt("perturb-vertex", 0.0)    # metres
PERTURB_GARMENT = _opt("perturb-garment", 0, int)
DUMP_POSITIONS = _opt("dump-positions", "", str)  # float64 .npy per frame stack

# --- scene constants (fixed: the benchmark's comparability depends on them) --
DT = 1.0 / 60.0
RPM = 40.0                       # ~5.3 m/s^2 centripetal at the bore: tumbling,
OMEGA = 2.0 * math.pi * RPM / 60.0   # not centrifuging (a < g)
EDGE_LEN = _opt("edge-len", 0.0062)  # target garment element size (m)

CLOTH_R_NOMINAL = 1.0e-3         # one-sided shell thickness (m), full = 2r
D_HAT_NOMINAL = 2.0e-3           # IPC activation distance (m)
CONTACT_RESISTANCE = 1.0e8
MU_DRUM_CLOTH = 0.4
MU_CLOTH_CLOTH = 0.3
# k_stretch = E * 2r = 1000 N/m, the terry-towel value of the cloth-machine
# calibrated material table.  Softer cloth (the 1e4 "legacy" value) crumples
# triangles to a quarter of their rest area, which drives the dynamic minimum
# triangle height below 2r + d_hat -- permanent self-contact, see --verify.
CLOTH_STRETCH_E = _opt("stretch-e", 5.0e5)
CLOTH_SHEAR_E = _opt("shear-e", 5.0e3)
CLOTH_BEND_E = _opt("bend-e", 1.0e5)
CLOTH_POISSON = 0.4
CLOTH_DENSITY = 200.0
CLOTH_STRAIN_RATE = 100.0
DRUM_KAPPA = 1.0e8
# The drum is kinematic, so its density is a *conditioning* parameter, not a
# physical one: a feather-light shell coupled to a heavy stiff affine body
# makes the relative PCG stopping criterion the drum's, and 2000 kg/m^3 cost
# ~28 % more PCG iterations and ~20 % more wall time for the same trajectory.
# 500 kg/m^3 still tracks the commanded angle to 0.04 deg (see --verify).
DRUM_DENSITY = _opt("drum-density", 5.0e2)
MOTOR_STRENGTH = _opt("motor-strength", 1.0e2)
TOL_RATE = _opt("tol-rate", 1e-3)

configure_benchmark_timers()
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Warn")))

from asset_dir import AssetDir  # noqa: E402  (kept next to the sibling samples)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

# --------------------------------------------------------------------------
# geometry: build the garments first -- d_hat and the shell thickness are
# capped by their element size (2r + d_hat must stay under 0.8 * the smallest
# triangle height or the cloth starts in permanent self-contact).
# --------------------------------------------------------------------------
spec = G.DrumSpec(radius=0.30, depth=0.40, wall_t=0.012,
                  n_fins=3, fin_height=0.06, fin_width=0.03, n_seg=48, n_z=10)

LAYER_MARGIN = 2.0e-3
_gap = 2 * CLOTH_R_NOMINAL + D_HAT_NOMINAL + LAYER_MARGIN

GARMENTS = [
    # name           builder                                             yaw    y
    ("towel",      lambda h: G.build_sheet(0.32, 0.22, h),                0.0, -0.230),
    ("pillowcase", lambda h: G.build_bag(0.28, 0.20, h, gap=_gap),       20.0, -0.208),
    ("shorts",     lambda h: G.build_trousers(0.20, 0.09, 0.24, 0.03, h, gap=_gap),
                                                                        -20.0, -0.186),
    ("washcloth",  lambda h: G.build_sheet(0.20, 0.20, h),               45.0, -0.164),
]

built = []
for _gi, (name, builder, yaw, y) in enumerate(GARMENTS):
    V, F = builder(EDGE_LEN)
    st = G.mesh_stats(V, F)
    if PERTURB_YAW != 0.0 and _gi == PERTURB_GARMENT:
        yaw = yaw + math.degrees(PERTURB_YAW)
    Vw = G.lay_flat(V, yaw, (0.0, y, 0.0))
    if PERTURB_VERTEX != 0.0 and _gi == PERTURB_GARMENT:
        Vw = Vw.copy()
        Vw[0, 0] += PERTURB_VERTEX
    reason = G.fits_in_bore(Vw, spec, clearance=0.008)
    if reason is not None:
        raise SystemExit(f"{name} does not fit in the drum bore: {reason}")
    built.append(dict(name=name, V=Vw, V_rest=V, F=F, stats=st))

MIN_TRI_HEIGHT = min(g["stats"]["min_tri_height"] for g in built)
# dataset rule: r <= 0.25 * h_min and 2r + d_hat <= 0.8 * h_min
CLOTH_R = min(CLOTH_R_NOMINAL, 0.25 * MIN_TRI_HEIGHT)
D_HAT = min(D_HAT_NOMINAL, 0.3 * MIN_TRI_HEIGHT)
if 2 * CLOTH_R + D_HAT > 0.8 * MIN_TRI_HEIGHT:
    raise SystemExit("contact-resolution rule violated; coarsen --edge-len")

drum_V, drum_F, drum_stats = G.make_drum(spec)

N_TRIS = int(sum(len(g["F"]) for g in built)) + int(len(drum_F))
N_VERTS = int(sum(len(g["V"]) for g in built)) + int(len(drum_V))

# --------------------------------------------------------------------------
config = Scene.default_config()
config["dt"] = DT
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = D_HAT
config["newton"]["velocity_tol"] = 0.05
config["linear_system"]["tol_rate"] = TOL_RATE
config["linear_system"]["fem_preconditioner"] = "mas"
scene = Scene(config)

ct = scene.contact_tabular()
ct.default_model(MU_CLOTH_CLOTH, CONTACT_RESISTANCE)
elem_drum = ct.create("drum")
elem_cloth = ct.create("cloth")
ct.insert(elem_drum, elem_cloth, MU_DRUM_CLOTH, CONTACT_RESISTANCE)
ct.insert(elem_cloth, elem_cloth, MU_CLOTH_CLOTH, CONTACT_RESISTANCE)
# the lifter boxes are welded *into* the bore wall, so the drum intersects
# itself by construction: it is one rigid body, self-contact is meaningless.
ct.insert(elem_drum, elem_drum, 0.0, CONTACT_RESISTANCE, False)

# --- drum ------------------------------------------------------------------
abd = AffineBodyConstitution()
stc = SoftTransformConstraint()
drum_mesh = trimesh(np.ascontiguousarray(drum_V), drum_F.astype(np.int32))
label_surface(drum_mesh)
abd.apply_to(drum_mesh, DRUM_KAPPA, DRUM_DENSITY)
stc.apply_to(drum_mesh, np.array([MOTOR_STRENGTH, MOTOR_STRENGTH], dtype=np.float64))
elem_drum.apply_to(drum_mesh)
drum_obj = scene.objects().create("drum")
drum_slot, _ = drum_obj.geometries().create(drum_mesh)


def commanded_transform(frame: int) -> np.ndarray:
    """Absolute pose commanded at the end of `frame`: R_z(omega * frame * dt)."""
    theta = OMEGA * DT * frame
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def drum_animation(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    view(geo.instances().find(builtin.is_constrained))[0] = 1
    view(geo.instances().find(builtin.aim_transform))[0] = commanded_transform(info.frame())


scene.animator().insert(drum_obj, drum_animation)

# --- bore end walls: implicit half-planes -----------------------------------
# A drum end cap is a flat disc normal to the rotation axis, so rotating it is
# geometrically a no-op.  Two half-planes close the bore exactly, with zero
# triangles and zero broad-phase cost; the contact/friction they provide is the
# same as a meshed cap's except that they do not drag tangentially.
ends_obj = scene.objects().create("bore_ends")
for sign in (+1.0, -1.0):
    plane = halfplane(np.array([0.0, 0.0, sign * spec.depth / 2]),
                      np.array([0.0, 0.0, -sign]))
    elem_drum.apply_to(plane)
    ends_obj.geometries().create(plane)

# --- garments ---------------------------------------------------------------
slbws = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()
stretch = ElasticModuli2D.youngs_poisson(CLOTH_STRETCH_E, CLOTH_POISSON)
shear = ElasticModuli2D.youngs_poisson(CLOTH_SHEAR_E, CLOTH_POISSON)

for g in built:
    mesh = trimesh(np.ascontiguousarray(g["V"]), g["F"].astype(np.int32))
    label_surface(mesh)
    slbws.apply_to(mesh, stretch_moduli=stretch, shear_moduli=shear,
                   mass_density=CLOTH_DENSITY, thickness=CLOTH_R,
                   strain_rate=CLOTH_STRAIN_RATE)
    dsb.apply_to(mesh, CLOTH_BEND_E, CLOTH_POISSON)
    elem_cloth.apply_to(mesh)
    obj = scene.objects().create(g["name"])
    slot, _ = obj.geometries().create(mesh)
    g["slot"] = slot
    g["gid"] = obj.geometries().ids()[0]

world.init(scene)
if not world.is_valid():
    raise SystemExit("world invalid after init -- sanity check failed")

if HEADLESS:
    print(f"tumbler: {len(built)} garments, {N_VERTS} verts, {N_TRIS} tris "
          f"(drum {len(drum_F)}), edge_len={EDGE_LEN * 1e3:.1f}mm, "
          f"h_min={MIN_TRI_HEIGHT * 1e3:.2f}mm, r={CLOTH_R * 1e3:.2f}mm, "
          f"d_hat={D_HAT * 1e3:.2f}mm, {RPM:g}rpm, {N_FRAMES} frames", flush=True)
    if PERTURB_YAW != 0.0 or PERTURB_VERTEX != 0.0:
        print(f"PERTURBED garment={PERTURB_GARMENT} "
              f"yaw={PERTURB_YAW:.3e}rad vertex_dx={PERTURB_VERTEX:.3e}m", flush=True)
    else:
        print("PERTURBED none (default scene)", flush=True)


# --------------------------------------------------------------------------
def cloth_positions():
    return [np.asarray(g["slot"].geometry().positions().view()).reshape(-1, 3)
            for g in built]


def drum_angle() -> float:
    A = np.asarray(drum_slot.geometry().transforms().view()[0]).reshape(4, 4)
    return math.atan2(A[1, 0], A[0, 0])


# --------------------------------------------------------------------------
if HEADLESS:
    import time

    audit = None
    if VERIFY:
        from verify import Audit
        audit = Audit(built, spec, DT, OMEGA, D_HAT, CLOTH_R)
        audit.observe(0, cloth_positions(), drum_angle(), None)

    frame_ms = []
    frame_stats = []
    traj = [np.vstack(cloth_positions()).astype(np.float64)] if DUMP_POSITIONS else None
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        world.advance()
        world.retrieve()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        frame_stats.append(snapshot_frame_stats(engine))
        if traj is not None:
            traj.append(np.vstack(cloth_positions()).astype(np.float64))
        if audit is not None:
            audit.observe(world.frame(), cloth_positions(), drum_angle(),
                          frame_stats[-1])
        elif world.frame() % 30 == 0:
            P = np.vstack(cloth_positions())
            print(f"track f{world.frame()} drum={math.degrees(drum_angle()):+8.2f}deg "
                  f"r_max={np.hypot(P[:, 0], P[:, 1]).max():.4f} "
                  f"y_mean={P[:, 1].mean():+.4f}", flush=True)

    P = np.vstack(cloth_positions())
    observables = {
        "final_frame": int(world.frame()),
        "n_tris": N_TRIS,
        "n_verts": N_VERTS,
        "edge_len": EDGE_LEN,
        "rpm": RPM,
        "cloth_max_radius": float(np.hypot(P[:, 0], P[:, 1]).max()),
        "cloth_mean_y": float(P[:, 1].mean()),
        "drum_angle_deg": float(math.degrees(drum_angle())),
    }
    if audit is not None:
        observables.update(audit.summary())
    if traj is not None:
        np.save(DUMP_POSITIONS, np.stack(traj))
        print(f"POSITIONS_DUMP {DUMP_POSITIONS} shape={np.stack(traj).shape}", flush=True)
    emit_benchmark_result(frame_ms, frame_stats, observables=observables)
    if audit is not None:
        audit.report()
    report_timers_if_enabled()
else:
    import polyscope as ps
    from polyscope import imgui
    from uipc.gui import SceneGUI

    sgui = SceneGUI(scene)
    ps.init()
    tri_surf, _, _ = sgui.register()
    tri_surf.set_edge_width(1.0)

    run = False

    def on_update():
        global run
        imgui.Text(f'frame: {world.frame()}')
        if imgui.Button('stop' if run else 'run'):
            run = not run
        if run:
            world.advance()
            world.retrieve()
            sgui.update()

    ps.set_user_callback(on_update)
    ps.show()
