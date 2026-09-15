"""Shared helpers for the showcase examples (96_flag .. 102_net).

Two lineages meet here.  The round-4 scenes (`100_mixer`, `101_press`,
`102_net`) and the round-6 cloth scenes (`96_flag`, `97_drape`, `98_rack`,
`99_funnel`) were built outside this repository around a private harness;
this module is that harness merged with the round-6 cloth toolkit, so the
scenes are self-contained in `examples/`:

  * CLI parsing shared by the scenes (frames, capture dir, output json)
  * per-frame benchmark statistics re-exported from the official
    `examples/benchmark_utils.py`, so the emitted format cannot drift
  * a capture writer producing the `f%05d.npz` + `meta.json` layout that the
    Warp offline renderer consumes, plus optional per-frame merged surface
    OBJ output
  * sanity-check helpers (uipc.core SanityChecker + NaN/explosion guards)
  * the tumbler benchmark's calibrated cloth material and its
    contact-resolution rule (`2r + d_hat <= 0.8 * min triangle height`), so
    every cloth scene's fidelity knob is one number, `--edge-len`
  * procedural closed trimeshes for affine bodies (boxes, cylinders, welded
    assemblies) and structured hex->tet solids for FEM bodies
  * `ClothObserver`: the per-frame physical audit in the spirit of the
    tumbler's verify.py -- finiteness, containment, inversion / collapse,
    bounded speed, "still moving" -- with a json summary

The garment builders live in `95_tumbler_garments/tumbler_geometry.py` and
are imported read-only; `G` is re-exported for the scenes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "95_tumbler_garments"))
import tumbler_geometry as G  # noqa: E402  (read-only import of the benchmark's builders)

from benchmark_utils import (  # noqa: E402  (the official emitter; same directory)
    configure_benchmark_timers,
    emit_benchmark_result,
    report_timers_if_enabled,
    snapshot_frame_stats,
)

__all__ = [
    "G", "add_r6_args", "build_argparser", "contact_resolution", "layer_gap",
    "ClothMaterial", "box_tri", "cylinder_tri", "weld", "tet_box", "tet_cylinder",
    "make_tetmesh", "make_abd_trimesh", "rot_z", "rot_axis", "fold_over_bar",
    "ClothObserver", "print_frame", "common_config", "finish",
    "Capture", "Runner", "world_positions", "transform4", "set_instance_transform",
    "run_sanity_check", "emit",
]

# --------------------------------------------------------------------------
# cloth material: the tumbler benchmark's calibrated terry-towel values
# --------------------------------------------------------------------------
CLOTH_STRETCH_E = 5.0e5
CLOTH_SHEAR_E = 5.0e3
CLOTH_BEND_E = 1.0e5
CLOTH_POISSON = 0.4
CLOTH_DENSITY = 200.0
CLOTH_STRAIN_RATE = 100.0
CLOTH_R_NOMINAL = 1.0e-3        # one-sided shell thickness (m)
D_HAT_NOMINAL = 2.0e-3          # IPC activation distance (m)
CONTACT_RESISTANCE = 1.0e8
DT = 1.0 / 60.0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_argparser(name: str, default_frames: int) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"libuipc showcase scene: {name}")
    ap.add_argument("frames", nargs="?", type=int, default=default_frames,
                    help=f"number of frames to simulate (default {default_frames})")
    ap.add_argument("--headless", action="store_true",
                    help="accepted for symmetry with the official samples; scenes are always headless")
    ap.add_argument("--capture", default=None,
                    help="directory for per-frame surface capture (npz + meta.json)")
    ap.add_argument("--capture-stride", type=int, default=1)
    ap.add_argument("--obj", default=None,
                    help="directory for per-frame merged surface OBJ output")
    ap.add_argument("--result", default=None, help="write the benchmark payload as json here")
    ap.add_argument("--tag", default="", help="free-form tag stored in the result json")
    ap.add_argument("--no-sanity", action="store_true", help="skip uipc sanity checks")
    ap.add_argument("--workspace", default=None, help="engine workspace directory")
    return ap


def add_r6_args(ap, default_edge_len):
    ap.add_argument("--edge-len", type=float, default=default_edge_len,
                    help=f"target cloth element size in metres (default {default_edge_len})")
    ap.add_argument("--tol-rate", type=float, default=None,
                    help="override linear_system/tol_rate")
    return ap


def contact_resolution(min_tri_height):
    """Dataset rule: r <= 0.25 h_min, d_hat <= 0.3 h_min, 2r + d_hat <= 0.8 h_min."""
    r = min(CLOTH_R_NOMINAL, 0.25 * min_tri_height)
    d_hat = min(D_HAT_NOMINAL, 0.3 * min_tri_height)
    if 2 * r + d_hat > 0.8 * min_tri_height:
        raise SystemExit(f"contact-resolution rule violated (h_min={min_tri_height*1e3:.2f} mm); "
                         "coarsen --edge-len")
    return r, d_hat


def layer_gap(r, d_hat, margin=2.0e-3):
    """Spacing between stacked cloth layers that starts *outside* contact."""
    return 2 * r + d_hat + margin


class ClothMaterial:
    """Applies the tumbler cloth to a trimesh (Baraff-Witkin strain limiting +
    discrete shell bending + contact element)."""

    def __init__(self, r, elem, stretch_e=CLOTH_STRETCH_E, shear_e=CLOTH_SHEAR_E,
                 bend_e=CLOTH_BEND_E, density=CLOTH_DENSITY):
        from uipc.constitution import (DiscreteShellBending, ElasticModuli2D,
                                       StrainLimitingBaraffWitkinShell)
        self.slbws = StrainLimitingBaraffWitkinShell()
        self.dsb = DiscreteShellBending()
        self.stretch = ElasticModuli2D.youngs_poisson(stretch_e, CLOTH_POISSON)
        self.shear = ElasticModuli2D.youngs_poisson(shear_e, CLOTH_POISSON)
        self.bend_e = bend_e
        self.density = density
        self.r = r
        self.elem = elem

    def make(self, V, F):
        from uipc.geometry import label_surface, trimesh
        mesh = trimesh(np.ascontiguousarray(np.asarray(V, dtype=np.float64)),
                       np.ascontiguousarray(np.asarray(F, dtype=np.int32)))
        label_surface(mesh)
        self.slbws.apply_to(mesh, stretch_moduli=self.stretch, shear_moduli=self.shear,
                            mass_density=self.density, thickness=self.r,
                            strain_rate=CLOTH_STRAIN_RATE)
        self.dsb.apply_to(mesh, self.bend_e, CLOTH_POISSON)
        self.elem.apply_to(mesh)
        return mesh


# --------------------------------------------------------------------------
# procedural geometry
# --------------------------------------------------------------------------
def _orient_outward(V, F):
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    if G.signed_volume(np.asarray(V, dtype=np.float64), F) < 0:
        F = F[:, ::-1].copy()
    return F


def box_tri(size, center=(0, 0, 0)):
    """Closed 12-triangle box, outward winding."""
    sx, sy, sz = (np.asarray(size, dtype=float) / 2.0)
    c = np.asarray(center, dtype=float)
    V = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]) + c
    # index = 4*ix + 2*iy + iz
    F = [
        (0, 1, 3), (0, 3, 2),       # -x
        (4, 6, 7), (4, 7, 5),       # +x
        (0, 4, 5), (0, 5, 1),       # -y
        (2, 3, 7), (2, 7, 6),       # +y
        (0, 2, 6), (0, 6, 4),       # -z
        (1, 5, 7), (1, 7, 3),       # +z
    ]
    return V, _orient_outward(V, F)


def cylinder_tri(R, L, axis="z", center=(0, 0, 0), n_seg=32, n_len=4):
    """Closed cylinder (tube + fan caps), outward winding."""
    V, F = [], []

    def add(p):
        V.append([float(p[0]), float(p[1]), float(p[2])])
        return len(V) - 1

    zs = np.linspace(-L / 2, L / 2, n_len + 1)
    rings = [[add((R * math.cos(2 * math.pi * j / n_seg), R * math.sin(2 * math.pi * j / n_seg), z))
              for j in range(n_seg)] for z in zs]
    for k in range(n_len):
        F += G.strip_between_loops(rings[k], rings[k + 1], closed=True)
    for ring, z in ((rings[0], zs[0]), (rings[-1], zs[-1])):
        c = add((0.0, 0.0, z))
        F += G.fan(c, ring, closed=True)
    V = np.array(V)
    if axis == "x":
        V = V[:, [2, 1, 0]] * np.array([1.0, 1.0, -1.0])
    elif axis == "y":
        V = V[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
    V = V + np.asarray(center, dtype=float)
    return V, _orient_outward(V, F)


def weld(parts):
    """Concatenate (V, F) parts into one mesh (components may intersect; use a
    contact element with self-contact disabled, as the tumbler drum does)."""
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def hex_to_tets(nx, ny, nz, vid):
    """Freudenthal / Kuhn 6-tet subdivision of every hex cell of a structured
    grid; conforming across shared faces because every cell uses the same
    main diagonal."""
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    T = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                base = np.array([i, j, k])
                for p in perms:
                    a = base.copy()
                    ids = [vid(*a)]
                    for ax in p:
                        a = a.copy()
                        a[ax] += 1
                        ids.append(vid(*a))
                    T.append(ids)
    return np.asarray(T, dtype=np.int64)


def _fix_tet_orientation(V, T):
    a, b, c, d = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]], V[T[:, 3]]
    vol = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
    T = T.copy()
    neg = vol < 0
    T[neg, 2], T[neg, 3] = T[neg, 3], T[neg, 2].copy()
    return T


def tet_box(size, n, center=(0, 0, 0)):
    """Structured tet mesh of a box; n = cells per axis (int or 3-tuple)."""
    nx, ny, nz = (n, n, n) if isinstance(n, int) else n
    sx, sy, sz = np.asarray(size, dtype=float)
    xs, ys, zs = (np.linspace(-sx / 2, sx / 2, nx + 1), np.linspace(-sy / 2, sy / 2, ny + 1),
                  np.linspace(-sz / 2, sz / 2, nz + 1))
    V = np.array([[x, y, z] for x in xs for y in ys for z in zs]) + np.asarray(center, float)
    vid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    T = hex_to_tets(nx, ny, nz, vid)
    return V, _fix_tet_orientation(V, T)


def tet_cylinder(R, L, n_disk, n_len, axis="z", center=(0, 0, 0)):
    """Structured tet mesh of a solid cylinder: a square grid mapped onto the
    disk (elliptical-grid mapping, no degenerate centre) and extruded."""
    u = np.linspace(-1, 1, n_disk + 1)
    zs = np.linspace(-L / 2, L / 2, n_len + 1)
    V = []
    for uu in u:
        for vv in u:
            x = uu * math.sqrt(1.0 - vv * vv / 2.0) * R
            y = vv * math.sqrt(1.0 - uu * uu / 2.0) * R
            for z in zs:
                V.append([x, y, z])
    V = np.asarray(V, dtype=np.float64)
    vid = lambda i, j, k: (i * (n_disk + 1) + j) * (n_len + 1) + k
    T = hex_to_tets(n_disk, n_disk, n_len, vid)
    if axis == "x":
        V = V[:, [2, 1, 0]] * np.array([1.0, 1.0, -1.0])
    elif axis == "y":
        V = V[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
    V = V + np.asarray(center, dtype=float)
    return V, _fix_tet_orientation(V, T)


def make_tetmesh(V, T):
    from uipc.geometry import (flip_inward_triangles, label_surface, label_triangle_orient,
                               tetmesh)
    m = tetmesh(np.ascontiguousarray(np.asarray(V, dtype=np.float64)),
                np.ascontiguousarray(np.asarray(T, dtype=np.int32)))
    label_surface(m)
    label_triangle_orient(m)
    return flip_inward_triangles(m)


def make_abd_trimesh(V, F, abd, kappa, density, elem, fixed=False, transform=None):
    """Closed trimesh -> affine body (the tumbler drum idiom)."""
    from uipc import builtin, view
    from uipc.geometry import label_surface, trimesh
    m = trimesh(np.ascontiguousarray(np.asarray(V, dtype=np.float64)),
                np.ascontiguousarray(np.asarray(F, dtype=np.int32)))
    label_surface(m)
    abd.apply_to(m, kappa, density)
    elem.apply_to(m)
    m.instances().resize(1)
    if transform is not None:
        set_instance_transform(m, 0, transform)
    view(m.instances().find(builtin.is_fixed))[0] = 1 if fixed else 0
    return m


def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def rot_axis(axis, theta):
    return transform4(rotate=(axis, theta))


def fold_over_bar(V, R_fold, bar_center, bar_dir, hang_dir):
    """Map a garment built flat in the x-y plane (x = fold direction, y along
    the bar, z = layer offset) onto a bar: an arc of radius R_fold + z over the
    top, then straight down on both sides.  Rest-length preserving in x."""
    P = np.asarray(V, dtype=np.float64)
    s = P[:, 0] - 0.5 * (P[:, 0].min() + P[:, 0].max())
    w = P[:, 1] - 0.5 * (P[:, 1].min() + P[:, 1].max())
    z = P[:, 2]
    rf = R_fold + z
    # The arc angle is taken from the *mid* radius for every layer, so the
    # layers of a two-layer garment stay parallel (same height at the same
    # build coordinate) instead of shearing past each other by pi/2 * gap at
    # the seams -- that shear put seam wedges inside the contact thickness.
    # The price is a +-gap/(2 R_fold) strain along the fold, which the shell
    # relaxes in the first frames.
    quarter = 0.5 * math.pi * R_fold
    on_arc = np.abs(s) <= quarter
    phi = np.where(on_arc, s / R_fold, np.sign(s) * 0.5 * math.pi)
    hang = np.where(on_arc, 0.0, np.abs(s) - quarter)
    u = rf * np.sin(phi)                    # across the bar
    y = rf * np.cos(phi) - hang             # up
    bar_dir = np.asarray(bar_dir, float) / np.linalg.norm(bar_dir)
    hang_dir = np.asarray(hang_dir, float) / np.linalg.norm(hang_dir)
    across = np.cross(bar_dir, hang_dir)
    across /= np.linalg.norm(across)
    Q = (np.asarray(bar_center, float)[None, :] + u[:, None] * across[None, :]
         + w[:, None] * bar_dir[None, :] + y[:, None] * hang_dir[None, :])
    return Q


# --------------------------------------------------------------------------
# per-frame physical audit
# --------------------------------------------------------------------------
class ClothObserver:
    """verify.py-style audit of one or more cloth geometries.

    bounds: dict of optional domain limits {"y_min", "y_max", "abs_x_max",
    "abs_z_max", "radius_max"} evaluated against every cloth vertex.
    """

    def __init__(self, scene, cloths, dt, r, d_hat, bounds=None, ground_y=None):
        """cloths: list of (name, geo_id, V_rest, F)."""
        self.scene = scene
        self.dt = dt
        self.skin = r + d_hat
        self.bounds = dict(bounds or {})
        self.ground_y = ground_y
        self.names = [c[0] for c in cloths]
        self.gids = [c[1] for c in cloths]
        self.F = [np.asarray(c[3], dtype=np.int64).reshape(-1, 3) for c in cloths]
        self.rest_area = [self._areas(np.asarray(c[2], float), f) for c, f in zip(cloths, self.F)]
        self.vmass = []
        for c, f, a0 in zip(cloths, self.F, self.rest_area):
            m = np.zeros(len(np.asarray(c[2])), dtype=np.float64)
            for k in range(3):
                np.add.at(m, f[:, k], a0 / 3.0)
            self.vmass.append(m * CLOTH_DENSITY * 2 * r)     # kg (areal density x area)
        self.mass_all = np.concatenate(self.vmass)
        self.prev = None
        self.rows = []
        self.n_tris = int(sum(len(f) for f in self.F))
        self.n_verts = int(sum(len(c[2]) for c in cloths))

    @staticmethod
    def _areas(V, F):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    @staticmethod
    def _min_height(V, F):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        e = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                      np.linalg.norm(a - c, axis=1)], axis=1)
        return float((2.0 * area[:, None] / np.maximum(e, 1e-30)).min())

    def positions(self):
        return [world_positions(self.scene, gid) for gid in self.gids]

    def observe(self, frame, extra=None):
        pos = self.positions()
        P = np.vstack(pos)
        row = {
            "frame": int(frame),
            "finite": bool(np.all(np.isfinite(P))),
            "y_min": float(P[:, 1].min()), "y_max": float(P[:, 1].max()),
            "abs_x_max": float(np.abs(P[:, 0]).max()),
            "abs_z_max": float(np.abs(P[:, 2]).max()),
            "radius_max": float(np.hypot(P[:, 0], P[:, 2]).max()),
            "centroid": P.mean(axis=0).tolist(),
        }
        ratio_lo, ratio_hi, h_min = math.inf, 0.0, math.inf
        for V, F, A0 in zip(pos, self.F, self.rest_area):
            R = self._areas(V, F) / A0
            ratio_lo = min(ratio_lo, float(R.min()))
            ratio_hi = max(ratio_hi, float(R.max()))
            h_min = min(h_min, self._min_height(V, F))
        row["area_ratio_min"] = ratio_lo
        row["area_ratio_max"] = ratio_hi
        row["tri_height_min"] = h_min
        if self.prev is not None:
            d = np.linalg.norm(P - self.prev, axis=1)
            dv = (P - self.prev) / self.dt
            row["mean_disp"] = float(d.mean())
            row["max_speed"] = float(d.max() / self.dt)
            row["ke"] = float(0.5 * np.dot(self.mass_all, (dv * dv).sum(axis=1)))
        else:
            row["mean_disp"] = row["max_speed"] = row["ke"] = 0.0
        self.prev = P
        if extra:
            row.update(extra)
        self.rows.append(row)
        return row

    def summary(self, moving_mm_per_frame=0.05):
        rows = self.rows
        run = rows[1:] if len(rows) > 1 else rows
        q = max(1, len(run) // 4)
        out = {
            "cloth_n_tris": self.n_tris,
            "cloth_n_verts": self.n_verts,
            "verify_all_finite": bool(all(x["finite"] for x in rows)),
            "verify_y_min": min(x["y_min"] for x in rows),
            "verify_y_max": max(x["y_max"] for x in rows),
            "verify_abs_x_max": max(x["abs_x_max"] for x in rows),
            "verify_abs_z_max": max(x["abs_z_max"] for x in rows),
            "verify_radius_max": max(x["radius_max"] for x in rows),
            "verify_area_ratio_min": min(x["area_ratio_min"] for x in rows),
            "verify_area_ratio_max": max(x["area_ratio_max"] for x in rows),
            "verify_tri_height_min": min(x["tri_height_min"] for x in rows),
            "verify_tri_height_floor": 2 * self.skin * 0.4,
            "verify_max_speed": max(x["max_speed"] for x in rows),
            "verify_mean_disp_mm": float(np.mean([x["mean_disp"] for x in run]) * 1e3),
            "verify_mean_disp_mm_last_quarter": float(np.mean([x["mean_disp"] for x in run[-q:]]) * 1e3),
            "verify_ke_max": float(max(x["ke"] for x in rows)),
            "verify_ke_last_quarter": float(np.mean([x["ke"] for x in run[-q:]])),
        }
        checks = {
            "finite": out["verify_all_finite"],
            "no_inversion_or_collapse": (out["verify_area_ratio_min"] > 0.05
                                         and out["verify_area_ratio_max"] < 20.0
                                         and out["verify_tri_height_min"] > out["verify_tri_height_floor"]),
            "bounded_speed": out["verify_max_speed"] < 50.0,
            "still_moving": out["verify_mean_disp_mm_last_quarter"] > moving_mm_per_frame,
        }
        if self.ground_y is not None:
            checks["above_ground"] = out["verify_y_min"] >= self.ground_y - self.skin
            out["verify_ground_y"] = self.ground_y
        for key, lim in self.bounds.items():
            if key == "y_min":
                checks["domain_y_min"] = out["verify_y_min"] >= lim - self.skin
            elif key == "y_max":
                checks["domain_y_max"] = out["verify_y_max"] <= lim + self.skin
            elif key == "abs_x_max":
                checks["domain_x"] = out["verify_abs_x_max"] <= lim + self.skin
            elif key == "abs_z_max":
                checks["domain_z"] = out["verify_abs_z_max"] <= lim + self.skin
            elif key == "radius_max":
                checks["domain_radius"] = out["verify_radius_max"] <= lim + self.skin
            out[f"verify_bound_{key}"] = lim
        out["checks"] = checks
        out["verify_ok"] = bool(all(checks.values()))
        return out


def print_frame(i, ms, s, extra=""):
    print(f"frame {i:4d} {ms:7.1f}ms  newton={s['newton_iterations']:2d} "
          f"pcg={s['linear_solver_iterations']:5d} ls={s['line_search_trials']:2d} "
          f"conv={s['converged']} {extra}", flush=True)


def common_config(d_hat, tol_rate=1e-3, mas=True):
    from uipc.core import Scene
    config = Scene.default_config()
    config["dt"] = DT
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["contact"]["d_hat"] = d_hat
    config["linear_system"]["tol_rate"] = tol_rate
    if mas:
        config["linear_system"]["fem_preconditioner"] = "mas"
    return config


# --------------------------------------------------------------------------
# capture writer
# --------------------------------------------------------------------------
class Capture:
    """Writes the per-frame surface capture consumed by the offline renderer."""

    def __init__(self, directory, scene, *, up_dir="y_up", ground_height=None,
                 ground_mode="manual", stride=1, radius=None, obj_dir=None):
        self.dir = Path(directory) if directory else None
        self.obj_dir = Path(obj_dir) if obj_dir else None
        self.stride = max(1, int(stride))
        self.scene = scene
        self.index = 0
        self.meta = {
            "up_dir": up_dir,
            "ground_height": ground_height,
            "ground_mode": ground_mode if ground_height is not None else "none",
            "radius": dict(radius or {}),
            "frame_ms": [],
        }
        self._parts = []           # (name, slot, kind)
        self._scene_io = None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
        if self.obj_dir:
            self.obj_dir.mkdir(parents=True, exist_ok=True)

    def bind(self, parts):
        """parts: list of (part_name, geometry_slot)."""
        self._parts = list(parts)

    @property
    def enabled(self):
        return self.dir is not None or self.obj_dir is not None

    @staticmethod
    def _world_geo(slot):
        from uipc.geometry import apply_transform, merge
        geo = slot.geometry()
        if geo.instances().size() >= 1:
            return merge(apply_transform(geo))
        return geo

    def _part_arrays(self, name, slot):
        geo = self._world_geo(slot)
        V = np.asarray(geo.positions().view()).reshape(-1, 3).astype(np.float32)
        dim = geo.dim()
        if dim == 3:
            T = np.asarray(geo.tetrahedra().topo().view()).reshape(-1, 4).astype(np.int32)
            kind = ""
        elif dim == 2:
            T = np.asarray(geo.triangles().topo().view()).reshape(-1, 3).astype(np.int32)
            kind = "surf"
        elif dim == 1:
            T = np.asarray(geo.edges().topo().view()).reshape(-1, 2).astype(np.int32)
            kind = "line"
        else:
            T = None
            kind = "point"
        return V, T, kind

    def write_frame(self, frame_index, force=False):
        if not self.enabled:
            return
        if not force and (frame_index % self.stride):
            return
        if self.dir:
            arrays = {}
            for name, slot in self._parts:
                V, T, kind = self._part_arrays(name, slot)
                arrays[f"{name}|kind"] = np.asarray(kind)
                arrays[f"{name}|V"] = V
                if T is not None and len(T):
                    arrays[f"{name}|T"] = T
            np.savez_compressed(self.dir / f"f{self.index:05d}.npz", **arrays)
        if self.obj_dir:
            from uipc.core import SceneIO
            if self._scene_io is None:
                self._scene_io = SceneIO(self.scene)
            self._scene_io.write_surface(str(self.obj_dir / f"surface_{self.index:05d}.obj"))
        self.index += 1

    def finish(self, frame_ms):
        if not self.dir:
            return
        self.meta["frame_ms"] = [float(v) for v in frame_ms]
        self.meta["kinds"] = {}
        for name, slot in self._parts:
            try:
                self.meta["kinds"][name] = self._part_arrays(name, slot)[2]
            except Exception:
                pass
        with open(self.dir / "meta.json", "w") as fp:
            json.dump(self.meta, fp, indent=1)


# --------------------------------------------------------------------------
# runner + result emission
# --------------------------------------------------------------------------
class Runner:
    """Drives the world, records the official per-frame statistics."""

    def __init__(self, world, engine, scene, args, scene_name):
        self.world = world
        self.engine = engine
        self.scene = scene
        self.args = args
        self.scene_name = scene_name
        self.frame_ms = []
        self.frame_stats = []
        self.trace = []          # per-frame scene-specific scalars
        self.capture = None

    def step(self, on_frame=None):
        t0 = time.perf_counter()
        self.world.advance()
        self.world.retrieve()
        ms = (time.perf_counter() - t0) * 1e3
        self.frame_ms.append(ms)
        self.frame_stats.append(snapshot_frame_stats(self.engine))
        if on_frame is not None:
            self.trace.append(on_frame())
        return ms

    # -- convergence / health -------------------------------------------------
    def health(self):
        bad = {
            "non_converged_frames": [i for i, s in enumerate(self.frame_stats)
                                     if not s.get("converged", False)],
            "newton_limit_frames": [i for i, s in enumerate(self.frame_stats)
                                    if s.get("hit_newton_limit", False)],
            "line_search_limit_frames": [i for i, s in enumerate(self.frame_stats)
                                         if s.get("hit_line_search_limit", False)],
            "incomplete_frames": [i for i, s in enumerate(self.frame_stats)
                                  if not s.get("completed", False)],
        }
        bad["ok"] = not any(bad[k] for k in list(bad))
        return bad

    def iteration_summary(self):
        def total(key):
            return int(sum(int(s.get(key, 0)) for s in self.frame_stats))
        n = max(1, len(self.frame_stats))
        return {
            "newton_total": total("newton_iterations"),
            "newton_per_frame": total("newton_iterations") / n,
            "newton_max": max((int(s.get("newton_iterations", 0)) for s in self.frame_stats), default=0),
            "line_search_total": total("line_search_trials"),
            "line_search_per_frame": total("line_search_trials") / n,
            "pcg_total": total("linear_solver_iterations"),
            "pcg_per_frame": total("linear_solver_iterations") / n,
        }

    def timing_summary(self):
        ms = self.frame_ms
        return {
            "frames": len(ms),
            "mean_ms": statistics.mean(ms),
            "median_ms": statistics.median(ms),
            "min_ms": min(ms),
            "max_ms": max(ms),
            "p95_ms": float(np.percentile(np.asarray(ms), 95)),
            "total_s": sum(ms) / 1e3,
        }


def world_positions(scene, geo_id):
    """World-space vertex positions of a geometry (instance transform applied)."""
    from uipc.geometry import apply_transform, merge
    slot, _ = scene.geometries().find(geo_id)
    geo = slot.geometry()
    if geo.instances().size() >= 1:
        geo = merge(apply_transform(geo))
    return np.asarray(geo.positions().view()).reshape(-1, 3).copy()


def run_sanity_check(world, label="sanity"):
    """Run the engine's own sanity checkers; return a json-able verdict."""
    from uipc.core import SanityCheckResult
    try:
        checker = world.sanity_checker()
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    try:
        result = checker.check()
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    out = {"available": True, "result": str(result), "errors": {}, "warns": {}, "infos": {}}
    for bucket in ("errors", "warns", "infos"):
        try:
            msgs = getattr(checker, bucket)()
        except Exception:
            continue
        acc = {}
        try:
            items = msgs.items()
        except AttributeError:
            items = enumerate(msgs)
        for key, msg in items:
            try:
                acc[str(key)] = {"name": msg.name(), "message": msg.message()}
            except Exception:
                acc[str(key)] = str(msg)
        out[bucket] = acc
    out["clean"] = (result == SanityCheckResult.Success) and not out["errors"]
    return out


def gpu_peak_mib():
    """Peak GPU memory of this process, via nvidia-smi sampling is unreliable;
    use the CUDA driver's own reporting if available."""
    try:
        import ctypes
        lib = ctypes.CDLL("libcudart.so")
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        if lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0:
            return (total.value - free.value) / (1024 ** 2)
    except Exception:
        pass
    return None


def emit(runner, observables, extra=None):
    payload = emit_benchmark_result(runner.frame_ms, runner.frame_stats, observables=observables)
    payload["scene"] = runner.scene_name
    payload["tag"] = runner.args.tag
    payload["timing"] = runner.timing_summary()
    payload["iterations"] = runner.iteration_summary()
    payload["health"] = runner.health()
    payload["gpu_mib_end"] = gpu_peak_mib()
    if extra:
        payload.update(extra)
    if runner.args.result:
        Path(runner.args.result).parent.mkdir(parents=True, exist_ok=True)
        with open(runner.args.result, "w") as fp:
            json.dump(payload, fp, indent=1)
        print(f"result json -> {runner.args.result}", flush=True)
    report_timers_if_enabled()
    return payload


# --------------------------------------------------------------------------
# small transform helpers
# --------------------------------------------------------------------------
def transform4(translate=(0, 0, 0), scale=1.0, rotate=None):
    """4x4 numpy homogeneous transform; rotate = (axis, angle_rad)."""
    M = np.eye(4)
    if rotate is not None:
        axis, angle = rotate
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        M[0:3, 0:3] = R
    s = np.asarray(scale, dtype=float)
    M[0:3, 0:3] = M[0:3, 0:3] * s
    M[0:3, 3] = np.asarray(translate, dtype=float)
    return M


def set_instance_transform(mesh, index, M):
    from uipc import Matrix4x4, view
    T = Matrix4x4.Identity()
    for r in range(4):
        for c in range(4):
            T[r, c] = float(M[r, c])
    view(mesh.transforms())[index] = T


# --------------------------------------------------------------------------
# standard scene epilogue: emit + capture meta + verify line
# --------------------------------------------------------------------------
def finish(runner, capture, observer, args, extra_obs, extra, init_sanity, final_sanity,
           edge_len, r, d_hat):
    capture.finish(runner.frame_ms)
    observables = {"frames": args.frames, "edge_len": edge_len, "cloth_r": r, "d_hat": d_hat}
    observables.update(observer.summary())
    observables.update(extra_obs)
    payload_extra = {
        "sanity": {"init": init_sanity, "final": final_sanity},
        "trace": observer.rows,
        "verify_checks": observables["checks"],
    }
    payload_extra.update(extra)
    print("VERIFY_RESULT " + json.dumps(
        {k: v for k, v in observables.items() if k.startswith("verify") or k == "checks"},
        separators=(",", ":"), default=float), flush=True)
    return emit(runner, observables, payload_extra)
