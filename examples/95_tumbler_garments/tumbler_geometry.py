"""Procedural geometry for the tumbler-garments benchmark (numpy only).

Self-contained port of the mesh builders used by the `cloth-machine` /
`cloth-dataset` projects, so the sample owns every vertex it simulates and
needs no binary assets:

  * ``make_drum``    -- horizontal-axis tumbler: a thick-walled open tube
                        (closed genus-1 surface, so the affine body has a
                        well-defined volume) with N radial lifter boxes welded
                        into the bore.  Adapted from
                        ``cloth_machine/garments.py::make_drum`` (which models a
                        washing-machine drum with an open front); here both bore
                        ends are closed by implicit half-planes in ``main.py``
                        instead of by meshed discs, which keeps the bore airtight
                        without spending triangles or broad-phase work on caps.
  * ``build_sheet``   -- flat towel / washcloth   (``dataset/garments.py::build_sheet``)
  * ``build_bag``     -- pillowcase: two rectangles welded on three sides
                        (``dataset/garments.py::build_bag``)
  * ``build_trousers``-- two U-panels welded at the side seams and the crotch
                        (``dataset/garments.py::build_trousers``)

The contact-resolution rule is the dataset's: libuipc activates contact below
``2r + d_hat`` and a mesh whose triangles are thinner than that starts in
permanent self-contact, so ``2r + d_hat <= 0.8 * min_triangle_height`` must hold
and stacked layers of a two-layer garment must sit ``layer_gap`` apart.

Everything is float64, deterministic and free of RNG except where a seed is
passed in explicitly.
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------
# small mesh helpers (ported from cloth_machine.garments)
# --------------------------------------------------------------------------


def _nseg(length: float, h: float, lo: int = 1) -> int:
    return max(lo, int(round(length / h)))


def strip_between_loops(loop_a, loop_b, closed: bool):
    """Triangulate the band between two equally sized vertex loops."""
    n = len(loop_a)
    assert n == len(loop_b)
    tris = []
    last = n if closed else n - 1
    for i in range(last):
        j = (i + 1) % n
        a0, a1, b0, b1 = loop_a[i], loop_a[j], loop_b[i], loop_b[j]
        tris += [(a0, a1, b1), (a0, b1, b0)]
    return tris


def fan(center: int, loop, closed: bool):
    n = len(loop)
    last = n if closed else n - 1
    return [(center, loop[i], loop[(i + 1) % n]) for i in range(last)]


def grid_quads(rows: int, cols: int, vid):
    """Two triangles per quad, diagonals alternating (no directional bias)."""
    tris = []
    for r in range(rows):
        for c in range(cols):
            a, b, cc, d = vid(r, c), vid(r, c + 1), vid(r + 1, c + 1), vid(r + 1, c)
            if (r + c) % 2 == 0:
                tris += [(a, b, cc), (a, cc, d)]
            else:
                tris += [(a, b, d), (b, cc, d)]
    return tris


def compact(V, F):
    """Drop unreferenced vertices and degenerate triangles."""
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
    F = F[keep]
    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return np.asarray(V, dtype=np.float64)[used], remap[F]


def signed_volume(V, F) -> float:
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", A, np.cross(B, C)).sum() / 6.0)


def mesh_stats(V, F) -> dict:
    """Edge/quality statistics used by the fidelity checks and the report."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    U, counts = np.unique(E, axis=0, return_counts=True)
    lengths = np.linalg.norm(V[U[:, 0]] - V[U[:, 1]], axis=1)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    e = np.stack([np.linalg.norm(b - a, axis=1),
                  np.linalg.norm(c - b, axis=1),
                  np.linalg.norm(a - c, axis=1)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        heights = 2.0 * area[:, None] / e
    return {
        "n_verts": int(len(V)),
        "n_tris": int(len(F)),
        "n_edges": int(len(U)),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "min_edge": float(lengths.min()),
        "mean_edge": float(lengths.mean()),
        "max_edge": float(lengths.max()),
        "min_area": float(area.min()),
        "min_tri_height": float(np.nanmin(heights)),
    }


# --------------------------------------------------------------------------
# the drum
# --------------------------------------------------------------------------


class DrumSpec:
    """Horizontal-axis tumbler.  Axis = +z, gravity = -y (see main.py)."""

    def __init__(self, radius=0.30, depth=0.40, wall_t=0.012, n_fins=3,
                 fin_height=0.06, fin_width=0.03, fin_phase_deg=0.0,
                 n_seg=48, n_z=10):
        self.radius = radius        # bore radius
        self.depth = depth          # bore length along the axis
        self.wall_t = wall_t
        self.n_fins = n_fins
        self.fin_height = fin_height
        self.fin_width = fin_width
        self.fin_phase_deg = fin_phase_deg
        self.n_seg = n_seg
        self.n_z = n_z

    @property
    def z_back(self):
        return -self.depth / 2

    @property
    def z_front(self):
        return +self.depth / 2

    @property
    def fin_angles_deg(self):
        return tuple((90.0 + 360.0 / self.n_fins * i + self.fin_phase_deg) % 360.0
                     for i in range(self.n_fins))

    def bore_half_width(self, y: float) -> float:
        """Half width in x of the bore at height y (|y| < radius)."""
        return math.sqrt(max(self.radius ** 2 - y ** 2, 0.0))


def make_drum(spec: DrumSpec):
    """Thick-walled open tube + lifter boxes.  Returns (V, F, stats).

    The tube is a *closed* surface (topologically a torus): inner wall, outer
    wall and the two annular rims.  Its signed volume is the wall volume, which
    is what the affine-body constitution integrates for mass.  The lifter boxes
    are separate closed components welded into the bore; they intersect the
    wall, which is harmless because drum-drum contact is disabled in the scene.
    """
    R, t = spec.radius, spec.wall_t
    zb, zf = spec.z_back, spec.z_front
    n_seg, n_z = spec.n_seg, spec.n_z
    V, F = [], []

    def add(p):
        V.append([float(p[0]), float(p[1]), float(p[2])])
        return len(V) - 1

    # closed (z, r) profile of the wall cross-section, revolved about +z.
    prof = []
    prof += [(zb + (zf - zb) * k / n_z, R) for k in range(n_z + 1)]   # bore
    prof += [(zf, R + t)]                                            # front rim
    prof += [(zb, R + t)]                                            # outer wall
    # (the closing segment front-rim -> bore start is generated by the cycle)
    rings = []
    for (z, r) in prof:
        rings.append([add((r * math.cos(2 * math.pi * j / n_seg),
                           r * math.sin(2 * math.pi * j / n_seg), z))
                      for j in range(n_seg)])
    for i in range(len(rings)):
        F += strip_between_loops(rings[i], rings[(i + 1) % len(rings)], closed=True)
    n_shell_v, n_shell_f = len(V), len(F)

    # lifters: closed boxes, radial extent [R - fin_height, R + t/2]
    fin_z0, fin_z1 = zb + 0.01, zf - 0.01
    n_fz = _nseg(fin_z1 - fin_z0, (zf - zb) / n_z)
    n_fr = max(1, _nseg(spec.fin_height, 2 * math.pi * R / n_seg))
    r_lo, r_hi = R - spec.fin_height, R + t / 2
    w = spec.fin_width / 2
    rs = np.linspace(r_lo, r_hi, n_fr + 1)
    rect = [(r, -w) for r in rs] + [(r, +w) for r in rs[::-1]]
    for ang in spec.fin_angles_deg:
        a = math.radians(ang)
        rad = np.array([math.cos(a), math.sin(a), 0.0])
        tan = np.array([-math.sin(a), math.cos(a), 0.0])
        loops = []
        for z in np.linspace(fin_z0, fin_z1, n_fz + 1):
            loops.append([add(rr * rad + ss * tan + np.array([0.0, 0.0, z]))
                          for (rr, ss) in rect])
        for k in range(n_fz):
            F += strip_between_loops(loops[k], loops[k + 1], closed=True)
        rc = float(np.mean([q[0] for q in rect]))
        sc = float(np.mean([q[1] for q in rect]))
        for loop, z in ((loops[0], fin_z0), (loops[-1], fin_z1)):
            c = add(rc * rad + sc * tan + np.array([0.0, 0.0, z]))
            F += fan(c, loop, closed=True)

    V, F = compact(np.array(V, dtype=np.float64), np.array(F, dtype=np.int64))
    if signed_volume(V, F) < 0.0:                 # make the winding outward
        F = F[:, ::-1].copy()
    st = mesh_stats(V, F)
    st.update(n_seg=n_seg, n_z=n_z, n_shell_verts=n_shell_v, n_shell_tris=n_shell_f,
              wall_volume=signed_volume(V, F))
    return V, F, st


# --------------------------------------------------------------------------
# garments (ported from cloth-dataset/dataset/garments.py)
# --------------------------------------------------------------------------


def build_sheet(L, W, h, **_):
    """A flat rectangular towel in the x-y plane, centred at the origin."""
    nx, ny = max(1, int(round(L / h))), max(1, int(round(W / h)))
    xs = np.linspace(-L / 2, L / 2, nx + 1)
    ys = np.linspace(-W / 2, W / 2, ny + 1)
    V = np.array([[x, y, 0.0] for x in xs for y in ys], dtype=np.float64)
    F = grid_quads(nx, ny, lambda r, c: r * (ny + 1) + c)
    return compact(V, np.asarray(F, dtype=np.int64))


def build_bag(L, W, h, gap, **_):
    """Pillowcase: two L x W panels welded along three sides (open at +x)."""
    nx, ny = max(2, int(round(L / h))), max(2, int(round(W / h)))
    xs = np.linspace(-L / 2, L / 2, nx + 1)
    ys = np.linspace(-W / 2, W / 2, ny + 1)
    V, ids_f, ids_b = [], {}, {}

    def add(p):
        V.append(list(p))
        return len(V) - 1

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if (i == 0) or (j == 0) or (j == ny):          # closed end + two sides
                k = add((x, y, 0.0))
                ids_f[(i, j)] = ids_b[(i, j)] = k
            else:
                ids_f[(i, j)] = add((x, y, +gap / 2))
                ids_b[(i, j)] = add((x, y, -gap / 2))
    welded = {k for k in ids_f if ids_f[k] == ids_b[k]}

    def quads(ids):
        # never emit a triangle whose three corners are all welded: the other
        # panel would duplicate it and the edge would become non-manifold.
        tris = []
        for r_ in range(nx):
            for c_ in range(ny):
                a, b, cc, d = (r_, c_), (r_, c_ + 1), (r_ + 1, c_ + 1), (r_ + 1, c_)
                cand = [(a, b, cc), (a, cc, d)] if (r_ + c_) % 2 == 0 else [(a, b, d), (b, cc, d)]
                if any(all(v in welded for v in tri) for tri in cand):
                    cand = [(a, b, d), (b, cc, d)] if (r_ + c_) % 2 == 0 else [(a, b, cc), (a, cc, d)]
                tris += [tuple(ids[v] for v in tri) for tri in cand]
        return tris

    F = [(t[0], t[2], t[1]) for t in quads(ids_f)] + quads(ids_b)
    return compact(np.array(V, dtype=np.float64), np.asarray(F, dtype=np.int64))


def build_trousers(leg, waist, W, crotch, h, gap, **_):
    """Two U-shaped panels welded at the side seams and the crotch."""
    n = lambda length: max(1, int(np.ceil(length / h)))
    X = np.r_[np.linspace(-leg, 0, n(leg) + 1), np.linspace(0, waist, n(waist) + 1)[1:]]
    Y = np.r_[np.linspace(-W / 2, -crotch / 2, n((W - crotch) / 2) + 1),
              np.linspace(-crotch / 2, crotch / 2, n(crotch) + 1)[1:],
              np.linspace(crotch / 2, W / 2, n((W - crotch) / 2) + 1)[1:]]
    points, ids, tri = [], {}, []

    def vertex(i, j):
        if (i, j) not in ids:
            ids[(i, j)] = len(points)
            points.append([X[i], Y[j], 0.0])
        return ids[(i, j)]

    for i in range(len(X) - 1):
        for j in range(len(Y) - 1):
            if (X[i] + X[i + 1]) / 2 < 0 and abs((Y[j] + Y[j + 1]) / 2) < crotch / 2:
                continue                                   # the gap between the legs
            a, b, c, d = vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)
            tri += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    P = np.array(points, dtype=np.float64)
    T = np.array(tri, dtype=np.int64)
    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    U, C = np.unique(E, axis=0, return_counts=True)
    seam = set()
    for a, b in U[C == 1]:
        opening = np.allclose(P[[a, b], 0], waist) or np.allclose(P[[a, b], 0], -leg)
        if not opening:                                    # side seams and crotch
            seam.update([int(a), int(b)])
    V = P.copy()
    V[:, 2] = gap / 2
    V[list(seam), 2] = 0.0
    V = V.tolist()
    back = []
    for i, p in enumerate(P):
        if i in seam:
            back.append(i)
        else:
            back.append(len(V))
            V.append([p[0], p[1], -gap / 2])
    back = np.asarray(back, dtype=np.int64)
    F = np.vstack([T, back[T[:, ::-1]]])
    return compact(np.array(V, dtype=np.float64), F.astype(np.int64))


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def lay_flat(V, yaw_deg: float, center):
    """Rotate a garment built flat in x-y into the horizontal x-z plane, spin it
    about the vertical axis by `yaw_deg`, then translate it to `center`."""
    P = np.asarray(V, dtype=np.float64)
    P = P - 0.5 * (P.min(axis=0) + P.max(axis=0))          # centre the build
    Q = np.stack([P[:, 0], P[:, 2], P[:, 1]], axis=1)      # x-y plane -> x-z plane
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    return Q @ R.T + np.asarray(center, dtype=np.float64)


def fits_in_bore(V, spec: DrumSpec, clearance: float) -> str | None:
    """Return a human-readable reason if any vertex is outside the bore minus
    `clearance` (also checks the lifter boxes), else None."""
    P = np.asarray(V, dtype=np.float64)
    r = np.hypot(P[:, 0], P[:, 1])
    if r.max() > spec.radius - clearance:
        return f"radius {r.max():.4f} > {spec.radius - clearance:.4f}"
    if np.abs(P[:, 2]).max() > spec.depth / 2 - clearance:
        return f"|z| {np.abs(P[:, 2]).max():.4f} > {spec.depth / 2 - clearance:.4f}"
    for ang in spec.fin_angles_deg:
        a = math.radians(ang)
        rad = P[:, 0] * math.cos(a) + P[:, 1] * math.sin(a)
        tan = -P[:, 0] * math.sin(a) + P[:, 1] * math.cos(a)
        hit = (rad > spec.radius - spec.fin_height - clearance) & \
              (np.abs(tan) < spec.fin_width / 2 + clearance)
        if hit.any():
            return f"{int(hit.sum())} vertices inside the lifter at {ang:.0f} deg"
    return None
