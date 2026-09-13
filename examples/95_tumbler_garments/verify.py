"""Physical-soundness audit for the tumbler-garments benchmark.

Used by `main.py --headless --verify`.  Every frame it re-reads the retrieved
cloth positions and answers, with numbers rather than assertions:

  containment   no vertex leaves the bore (radius, axial extent, lifter boxes)
  regularity    no NaN/Inf, no collapsed or inverted triangle, bounded speed
  drive         the drum tracks its commanded absolute angle
  motion        the garments keep tumbling instead of settling into a heap
  solver        Newton / PCG counts per frame stay bounded and do not diverge

Nothing here touches the timed section of the run: `main.py` calls `observe()`
after the per-frame stopwatch has already been read.
"""
from __future__ import annotations

import json
import math
import os
import subprocess

import numpy as np


def gpu_memory_mib() -> int:
    """Device-wide memory in use, sampled from the main thread between frames
    (a background sampler racing the first advance()'s CUDA-graph capture has
    been seen to abort the backend)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        for line in out.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return int(mem)
    except Exception:
        pass
    return 0


class Audit:
    def __init__(self, built, spec, dt, omega, d_hat, cloth_r):
        self.spec = spec
        self.dt = dt
        self.omega = omega
        # a vertex is legitimately held one contact gap away from the wall
        self.skin = cloth_r + d_hat
        self.names = [g["name"] for g in built]
        self.F = [np.asarray(g["F"], dtype=np.int64).reshape(-1, 3) for g in built]
        self.rest_area = [self._areas(np.asarray(g["V_rest"]), f)
                          for g, f in zip(built, self.F)]
        # lumped vertex mass, in units of areal density: one third of the rest
        # area of each incident triangle.  Only ratios and A/B differences are
        # read off it, so the constant areal density is left out on purpose.
        self.vmass = []
        for g, f, a0 in zip(built, self.F, self.rest_area):
            m = np.zeros(len(np.asarray(g["V_rest"])), dtype=np.float64)
            for k in range(3):
                np.add.at(m, f[:, k], a0 / 3.0)
            self.vmass.append(m)
        self.mass_all = np.concatenate(self.vmass)
        # vertices that share a triangle are *supposed* to be close; exclude
        # them from the cloth-cloth proximity statistic.
        self.nbr_keys = self._neighbour_keys(
            built, self.F, int(sum(len(np.asarray(g["V_rest"])) for g in built)))
        self.n_seg = getattr(spec, "n_seg", 48)
        self.prev = None
        self.rows = []
        self.turn = [0.0] * len(built)     # unwrapped centroid angle travel
        self.last_phi = None
        self.gpu_mib = 0

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _neighbour_keys(built, Fs, n_total):
        """Sorted int64 keys i*n+j for every ordered vertex pair that shares a
        triangle (plus i==i), over the concatenated garment vertex block."""
        off, pairs = 0, []
        for g, F in zip(built, Fs):
            n = len(np.asarray(g["V_rest"]))
            T = F + off
            a = np.repeat(T, 3, axis=1).reshape(-1)          # i
            b = np.tile(T, (1, 3)).reshape(-1)               # j
            pairs.append(a.astype(np.int64) * n_total + b.astype(np.int64))
            off += n
        idx = np.arange(n_total, dtype=np.int64)
        pairs.append(idx * n_total + idx)
        return np.unique(np.concatenate(pairs))

    def _cloth_cloth_min_dist(self, P):
        """Smallest distance between two cloth vertices that do not share a
        triangle.  A *proxy* for the true point-triangle gap -- it bounds it
        from above -- but it is the same proxy in every arm, which is what the
        A/B needs.  Returns inf if scipy is unavailable."""
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return float("inf")
        n = P.shape[0]
        tree = cKDTree(P)
        d, j = tree.query(P, k=12, workers=-1)
        i = np.repeat(np.arange(n, dtype=np.int64)[:, None], d.shape[1], axis=1)
        keys = i * n + j.astype(np.int64)
        excluded = np.isin(keys.reshape(-1), self.nbr_keys).reshape(d.shape)
        dd = np.where(excluded | ~np.isfinite(d), np.inf, d)
        return float(dd.min())

    def _bore_gap_min(self, P):
        """Signed distance from every cloth vertex to the *polygonal* bore wall
        (n_seg flats at circumradius R), positive inside.  Exact for the tube;
        the lifters are covered by `lifter_depth`."""
        n = self.n_seg
        R = self.spec.radius
        r = np.hypot(P[:, 0], P[:, 1])
        th = np.arctan2(P[:, 1], P[:, 0])
        step = 2.0 * math.pi / n
        phi = np.remainder(th + step / 2.0, step) - step / 2.0
        return float((R * math.cos(step / 2.0) - r * np.cos(phi)).min())

    @staticmethod
    def _areas(V, F):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    @staticmethod
    def _min_height(V, F):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        e = np.stack([np.linalg.norm(b - a, axis=1),
                      np.linalg.norm(c - b, axis=1),
                      np.linalg.norm(a - c, axis=1)], axis=1)
        return float((2.0 * area[:, None] / np.maximum(e, 1e-30)).min())

    def _lifter_depth(self, P, drum_angle):
        """Max depth by which any vertex has entered a lifter box (<=0 = clear).
        The lifters turn with the drum, so the box frames are offset by the
        drum's *measured* angle, not by their build angle."""
        worst = -1.0
        for ang in self.spec.fin_angles_deg:
            a = math.radians(ang) + drum_angle
            rad = P[:, 0] * math.cos(a) + P[:, 1] * math.sin(a)
            tan = -P[:, 0] * math.sin(a) + P[:, 1] * math.cos(a)
            # signed distance to the box surface, positive inside
            d = np.minimum.reduce([
                rad - (self.spec.radius - self.spec.fin_height),
                self.spec.radius + self.spec.wall_t / 2 - rad,
                self.spec.fin_width / 2 - np.abs(tan),
                self.spec.depth / 2 - 0.01 - np.abs(P[:, 2])])
            worst = max(worst, float(d.max()))
        return worst

    # -- per frame ----------------------------------------------------------
    def observe(self, frame, positions, drum_angle, stats):
        P = np.vstack(positions)
        finite = bool(np.all(np.isfinite(P)))
        r = np.hypot(P[:, 0], P[:, 1])
        row = {
            "frame": int(frame),
            "finite": finite,
            "r_max": float(r.max()),
            "abs_z_max": float(np.abs(P[:, 2]).max()),
            "lifter_depth": self._lifter_depth(P, drum_angle),
            "drum_angle": float(drum_angle),
            "y_min": float(P[:, 1].min()),
            "y_max": float(P[:, 1].max()),
        }
        ratio_lo, ratio_hi, h_min = math.inf, 0.0, math.inf
        for V, F, A0 in zip(positions, self.F, self.rest_area):
            A = self._areas(V, F)
            ratio_lo = min(ratio_lo, float((A / A0).min()))
            ratio_hi = max(ratio_hi, float((A / A0).max()))
            h_min = min(h_min, self._min_height(V, F))
        row["area_ratio_min"] = ratio_lo
        row["area_ratio_max"] = ratio_hi
        row["tri_height_min"] = h_min
        row["cc_min_dist"] = self._cloth_cloth_min_dist(P)
        row["bore_gap_min"] = self._bore_gap_min(P)
        # energy, in units of areal density (see self.vmass): gravitational
        # potential now, kinetic from the finite-difference velocity below.
        row["pe_grav"] = float(9.8 * np.dot(self.mass_all, P[:, 1]))

        # motion: mean vertex speed and centroid angular travel about the axis
        if self.prev is not None:
            dv = (P - self.prev) / self.dt
            d = np.linalg.norm(P - self.prev, axis=1)
            row["mean_disp"] = float(d.mean())
            row["max_speed"] = float(d.max() / self.dt)
            row["ke"] = float(0.5 * np.dot(self.mass_all, (dv * dv).sum(axis=1)))
        else:
            row["mean_disp"] = 0.0
            row["max_speed"] = 0.0
            row["ke"] = 0.0
        row["e_tot"] = row["ke"] + row["pe_grav"]
        phi = np.array([math.atan2(V[:, 1].mean(), V[:, 0].mean()) for V in positions])
        if self.last_phi is not None:
            dphi = np.remainder(phi - self.last_phi + math.pi, 2 * math.pi) - math.pi
            self.turn = [t + float(x) for t, x in zip(self.turn, dphi)]
        self.last_phi = phi
        self.prev = P

        if frame % 10 == 0:
            self.gpu_mib = max(self.gpu_mib, gpu_memory_mib())
        if stats is not None:
            row["newton"] = int(stats.get("newton_iterations", -1))
            row["pcg"] = int(stats.get("linear_solver_iterations", -1))
            row["line_search"] = int(stats.get("line_search_trials", -1))
            row["converged"] = int(stats.get("converged", -1))
            row["hit_newton_limit"] = int(bool(stats.get("hit_newton_limit", 0)))
            row["hit_ls_limit"] = int(bool(stats.get("hit_line_search_limit", 0)))
            row["last_ccd_toi"] = float(stats.get("last_ccd_toi", 1.0))
            row["last_ls_alpha"] = float(stats.get("last_line_search_alpha", 1.0))
            row["last_cfl_alpha"] = float(stats.get("last_cfl_alpha", 1.0))
        self.rows.append(row)

    # -- summary ------------------------------------------------------------
    def summary(self) -> dict:
        rows = self.rows
        run = rows[1:] if len(rows) > 1 else rows
        ang = np.unwrap([x["drum_angle"] for x in rows])
        cmd = self.omega * self.dt * np.array([x["frame"] for x in rows])
        err = np.degrees(ang - ang[0] - cmd)
        bore_limit = self.spec.radius + self.skin
        newton = np.array([x["newton"] for x in run if "newton" in x])
        pcg = np.array([x["pcg"] for x in run if "pcg" in x])
        half = max(1, len(newton) // 4)
        out = {
            "verify_all_finite": bool(all(x["finite"] for x in rows)),
            "verify_r_max": max(x["r_max"] for x in rows),
            "verify_r_limit": bore_limit,
            "verify_contained_radial": bool(max(x["r_max"] for x in rows) <= bore_limit),
            "verify_abs_z_max": max(x["abs_z_max"] for x in rows),
            "verify_z_limit": self.spec.depth / 2 + self.skin,
            "verify_contained_axial": bool(
                max(x["abs_z_max"] for x in rows) <= self.spec.depth / 2 + self.skin),
            "verify_lifter_depth_max": max(x["lifter_depth"] for x in rows),
            "verify_area_ratio_min": min(x["area_ratio_min"] for x in rows),
            "verify_area_ratio_max": max(x["area_ratio_max"] for x in rows),
            "verify_tri_height_min": min(x["tri_height_min"] for x in rows),
            "verify_max_speed": max(x["max_speed"] for x in rows),
            "verify_drum_track_err_deg_max": float(np.max(np.abs(err))),
            "verify_drum_track_err_deg_final": float(err[-1]),
            "verify_drum_turn_deg": float(math.degrees(ang[-1] - ang[0])),
            "verify_drum_commanded_deg": float(math.degrees(cmd[-1])),
            "verify_mean_disp_mm": float(np.mean([x["mean_disp"] for x in run]) * 1e3),
            "verify_mean_disp_mm_last_quarter": float(
                np.mean([x["mean_disp"] for x in run[-half:]]) * 1e3),
            "verify_centroid_turn_deg": [float(math.degrees(t)) for t in self.turn],
            "verify_newton_min": int(newton.min()), "verify_newton_max": int(newton.max()),
            "verify_newton_mean": float(newton.mean()),
            "verify_newton_first_quarter": float(newton[:half].mean()),
            "verify_newton_last_quarter": float(newton[-half:].mean()),
            "verify_pcg_min": int(pcg.min()), "verify_pcg_max": int(pcg.max()),
            "verify_pcg_mean": float(pcg.mean()),
            "verify_pcg_first_quarter": float(pcg[:half].mean()),
            "verify_pcg_last_quarter": float(pcg[-half:].mean()),
            "verify_gpu_mem_proc_max_mib": int(self.gpu_mib),
            "verify_cc_min_dist": min(x["cc_min_dist"] for x in rows),
            "verify_bore_gap_min": min(x["bore_gap_min"] for x in rows),
            "verify_ke_mean": float(np.mean([x["ke"] for x in run])),
            "verify_ke_max": float(max(x["ke"] for x in rows)),
            "verify_e_tot_first": float(rows[1]["e_tot"]) if len(rows) > 1 else 0.0,
            "verify_e_tot_last": float(rows[-1]["e_tot"]),
            "verify_e_tot_max": float(max(x["e_tot"] for x in rows)),
            "verify_ccd_toi_min": float(min(x.get("last_ccd_toi", 1.0) for x in run)),
            "verify_ccd_toi_clamped_frames": int(
                sum(1 for x in run if x.get("last_ccd_toi", 1.0) < 1.0)),
            "verify_ls_alpha_min": float(min(x.get("last_ls_alpha", 1.0) for x in run)),
            "verify_ls_alpha_cut_frames": int(
                sum(1 for x in run if x.get("last_ls_alpha", 1.0) < 1.0)),
            "verify_cfl_alpha_min": float(min(x.get("last_cfl_alpha", 1.0) for x in run)),
            "verify_hit_newton_limit_frames": int(
                sum(x.get("hit_newton_limit", 0) for x in run)),
            "verify_hit_ls_limit_frames": int(sum(x.get("hit_ls_limit", 0) for x in run)),
            "verify_line_search_total": int(
                sum(x.get("line_search", 0) for x in run)),
            "verify_newton_total": int(newton.sum()),
            "verify_pcg_total": int(pcg.sum()),
            "verify_not_converged_frames": int(sum(1 for x in run if x.get("converged", 1) == 0)),
        }
        out["verify_ok"] = bool(
            out["verify_all_finite"] and out["verify_contained_radial"]
            and out["verify_contained_axial"] and out["verify_area_ratio_min"] > 0.05
            and out["verify_area_ratio_max"] < 20.0
            and out["verify_tri_height_min"] > 2 * self.skin * 0.4
            and out["verify_max_speed"] < 50.0
            and out["verify_drum_track_err_deg_max"] < 5.0
            and out["verify_mean_disp_mm_last_quarter"] > 0.05)
        return out

    def report(self):
        print("VERIFY_RESULT " + json.dumps(self.summary(), separators=(",", ":")), flush=True)
        print("VERIFY_TRACE " + json.dumps(self.rows, separators=(",", ":")), flush=True)
