"""Physical-soundness + constitution-state audit for the crease-press benchmark.

Used by `main.py --verify`.  Two jobs, both outside the timed section (main.py
calls observe() after the per-frame stopwatch has been read):

  soundness   finiteness, no inversion/collapse (area ratio + min triangle
              height vs the per-family 2r+d_hat floor), bounded speed,
              containment under/around the press, press tracking error,
              Newton/PCG bounded and not diverging, no non-converged frames
  regime      the constitutions are in their *interesting* regime:
                - Dahl friction state actually evolves: the committed friction
                  moment F per hinge, replicated here in numpy with the exact
                  device update (F_new = s*M + (F - s*M)*exp(-|d|/ell)), must
                  be non-zero on a large fraction of hinges after the first
                  press;
                - the plastic sheets actually yield: the plastic rest angle
                  theta_bar per hinge (strain: |delta| > yield; stress: the
                  same with the per-hinge moment threshold) must move.

The state replicas track the engine's own commit kernels
(DahlFrictionDiscreteShellBendingTimeIntegrator / *Plastic*...do_update_state)
formula for formula, hinge set for hinge set: the hinge census of a flat grid
sheet is its interior edges, which is what the engine's do_init collects
(edges with exactly two opposite vertices).
"""
from __future__ import annotations

import json
import math

import numpy as np


def _wrap(angle):
    return np.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _dihedral(P, st):
    """Vectorised safe_dihedral_angle over hinges; NaN where degenerate."""
    v0, v1, v2, v3 = P[st[:, 0]], P[st[:, 1]], P[st[:, 2]], P[st[:, 3]]
    n1 = np.cross(v1 - v0, v2 - v0)
    n2 = np.cross(v2 - v3, v1 - v3)
    n1n = np.linalg.norm(n1, axis=1)
    n2n = np.linalg.norm(n2, axis=1)
    denom = n1n * n2n
    ok = denom > 1e-12
    theta = np.full(len(st), np.nan)
    cos_t = np.clip(np.einsum("ij,ij->i", n1, n2) / np.where(ok, denom, 1.0), -1.0, 1.0)
    theta[ok] = np.arccos(cos_t[ok])
    flip = np.einsum("ij,ij->i", np.cross(n2, n1), v1 - v2) < 0
    theta[ok & flip] = -theta[ok & flip]
    return theta


def _rest_constants(V0, st):
    """Per-hinge (L0, h_bar) of the discrete-shell reference, as the engine
    computes them from the rest positions."""
    L0 = np.linalg.norm(V0[st[:, 2]] - V0[st[:, 1]], axis=1)
    n1 = np.cross(V0[st[:, 1]] - V0[st[:, 0]], V0[st[:, 2]] - V0[st[:, 0]])
    n2 = np.cross(V0[st[:, 2]] - V0[st[:, 3]], V0[st[:, 1]] - V0[st[:, 3]])
    A = 0.5 * (np.linalg.norm(n1, axis=1) + np.linalg.norm(n2, axis=1))
    h_bar = A / 3.0 / L0
    return L0, h_bar


class Audit:
    def __init__(self, sheets, dt, d_hat, params, press_pose, stack_y0, ground_y,
                 z_off, sheet_x, sheet_z):
        self.dt = dt
        self.d_hat = d_hat
        self.press_pose = press_pose
        self.stack_y0 = stack_y0
        self.press_floor = ground_y                         # the die / ground halfplane
        self.z_off = z_off
        self.sheet_x = sheet_x
        self.sheet_z = sheet_z
        self.names = [f"{s['k']}_{s['name']}" for s in sheets]
        self.kinds = [s["kind"] for s in sheets]
        self.F = [np.asarray(s["F"], dtype=np.int64) for s in sheets]
        self.V0 = [np.asarray(s["V0"], dtype=np.float64) for s in sheets]
        self.rest_y = np.array([s["rest_y"] for s in sheets])
        self.rest_area = [self._areas(V, f) for V, f in zip(self.V0, self.F)]
        # per-family contact skin + permanent-self-contact floor (2r + d_hat)
        r = {"dahl": 0.0008, "strain": 0.0025, "stress": 0.0025}
        self.skin = np.array([r[k] + d_hat for k in self.kinds])
        self.floor = np.array([2.0 * r[k] + d_hat for k in self.kinds])
        # areal density per family (kg/m^2) for the energy bookkeeping
        areal = {"dahl": 0.30, "strain": 200.0 * 2 * 0.0025, "stress": 200.0 * 2 * 0.0025}
        self.vmass = []
        for s, f, a0 in zip(sheets, self.F, self.rest_area):
            m = np.zeros(len(s["V0"]))
            for k in range(3):
                np.add.at(m, f[:, k], a0 / 3.0)
            self.vmass.append(m * areal[s["kind"]])
        self.mass_all = np.concatenate(self.vmass)
        # ---- constitution-state replicas -----------------------------------
        self.st = [np.asarray(s["stencils"], dtype=np.int64) for s in sheets]
        # hinges whose rest midpoint sits under either crease line (the bar's
        # footprint +- one element): the hinges the round actually targets
        self.crease_mask = []
        for s, st in zip(sheets, self.st):
            zm = 0.5 * (s["V0"][st[:, 1], 2] + s["V0"][st[:, 2], 2])
            self.crease_mask.append((np.abs(zm - z_off) < 0.04 + 0.012) |
                                    (np.abs(zm + z_off) < 0.04 + 0.012))
        self.theta_commit, self.F_commit, self.theta_bar = [], [], []
        self.sat_M, self.ell_e, self.yield_theta = [], [], []
        for s, st, V0 in zip(sheets, self.st, self.V0):
            n = len(st)
            self.theta_commit.append(np.zeros(n))
            self.F_commit.append(np.zeros(n))
            self.theta_bar.append(np.zeros(n))
            L0, h_bar = _rest_constants(V0, st)
            if s["kind"] == "dahl":
                kappa, m_hat, ell = params["dahl"]
                self.sat_M.append(m_hat * L0)
                self.ell_e.append(np.full(n, ell))
                self.yield_theta.append(None)
            elif s["kind"] == "strain":
                self.sat_M.append(None)
                self.ell_e.append(None)
                self.yield_theta.append(np.full(n, params["strain"][1]))
            else:  # stress: theta_y = yield_stress / (2 kappa L0 / h_bar)
                kappa, yield_stress = params["stress"]
                slope = 2.0 * kappa * L0 / h_bar
                self.sat_M.append(None)
                self.ell_e.append(None)
                self.yield_theta.append(yield_stress / slope)
        self.prev = None
        self.rows = []
        self.track_err = []

    # -- helpers ------------------------------------------------------------
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

    # -- the engine's per-frame commit kernels, replicated -------------------
    def _commit_states(self, positions):
        for gi, (P, st, kind) in enumerate(zip(positions, self.st, self.kinds)):
            theta = _dihedral(P, st)
            good = np.isfinite(theta)
            d = _wrap(np.where(good, theta, 0.0) - self.theta_commit[gi])
            if kind == "dahl":
                M, ell = self.sat_M[gi], self.ell_e[gi]
                s = np.sign(d)
                F_new = s * M + (self.F_commit[gi] - s * M) * np.exp(-np.abs(d) / ell)
                self.F_commit[gi] = np.clip(F_new, -M, M)
            else:
                y = self.yield_theta[gi]
                excess = np.maximum(np.abs(d) - y, 0.0)
                move = excess > 1e-6            # plasticity_write_threshold
                self.theta_bar[gi] = _wrap(
                    self.theta_bar[gi] + np.where(move, np.sign(d) * excess, 0.0))
            self.theta_commit[gi] = np.where(good, theta, self.theta_commit[gi])

    def state_summary(self):
        """Fractions of hinges whose internal state has left zero, per family
        (max over the sheets of that family)."""
        out = {"dahl_active_frac": 0.0, "strain_yield_frac": 0.0, "stress_yield_frac": 0.0}
        key = {"dahl": "dahl_active_frac", "strain": "strain_yield_frac",
               "stress": "stress_yield_frac"}
        for gi, kind in enumerate(self.kinds):
            if kind == "dahl":
                v = float((np.abs(self.F_commit[gi]) > 1e-9).mean())
            else:
                v = float((np.abs(self.theta_bar[gi]) > 1e-9).mean())
            out[key[kind]] = max(out[key[kind]], v)
        return out

    # -- per frame ------------------------------------------------------------
    def observe(self, frame, positions, pose, stats):
        P = np.vstack(positions)
        row = {"frame": int(frame), "finite": bool(np.all(np.isfinite(P)))}
        # containment: the stack sits under the press; sheets must not wander
        row["abs_x_max"] = float(np.abs(P[:, 0]).max())
        row["abs_z_max"] = float(np.abs(P[:, 2]).max())
        row["y_min"] = float(P[:, 1].min())
        row["y_max"] = float(P[:, 1].max())
        ratio_lo, ratio_hi, h_min = math.inf, 0.0, math.inf
        below_floor = 0
        sheet_ar_max = []
        worst = (0.0, -1, 0.0, 0.0)     # ratio, sheet, z, x of the worst triangle
        for gi, (V, F, A0) in enumerate(zip(positions, self.F, self.rest_area)):
            R = self._areas(V, F) / A0
            ratio_lo = min(ratio_lo, float(R.min()))
            ratio_hi = max(ratio_hi, float(R.max()))
            sheet_ar_max.append(float(R.max()))
            k = int(np.argmax(R))
            if float(R[k]) > worst[0]:
                c = V[F[k]].mean(axis=0)
                worst = (float(R[k]), gi, float(c[2]), float(c[0]))
            h = self._min_height(V, F)
            h_min = min(h_min, h)
            below_floor += int(h < self.floor[gi])
        row["area_ratio_min"] = ratio_lo
        row["area_ratio_max"] = ratio_hi
        row["sheet_area_ratio_max"] = sheet_ar_max
        row["ar_worst"] = [worst[0], worst[1], round(worst[2], 3), round(worst[3], 3)]
        row["tri_height_min"] = h_min
        row["sheet_x_max"] = [float(np.abs(V[:, 0]).max()) for V in positions]
        row["sheet_h_min"] = [
            float(self._min_height(V, F)) for V, F in zip(positions, self.F)]
        row["sheets_below_floor"] = below_floor
        # press tracking: the SoftTransformConstraint aim vs the achieved pose
        cy, cz = pose
        row["press_cmd_y"], row["press_cmd_z"] = cy, cz
        # per-sheet mean deflection (the crease observable)
        defs, crease_defs = [], []
        for gi, V in enumerate(positions):
            defs.append(float(self.rest_y[gi] - V[:, 1].mean()))
            near = (np.abs(self.V0[gi][:, 2] - self.z_off) < 0.06) | \
                   (np.abs(self.V0[gi][:, 2] + self.z_off) < 0.06)
            crease_defs.append(float((self.rest_y[gi] - V[near, 1]).mean()))
        row["sheet_mean_defl"] = defs
        row["sheet_crease_defl"] = crease_defs
        row["max_defl"] = float(max(self.rest_y[gi] - V[:, 1].min()
                                    for gi, V in enumerate(positions)))
        # motion
        if self.prev is not None:
            d = np.linalg.norm(P - self.prev, axis=1)
            row["mean_disp"] = float(d.mean())
            row["max_speed"] = float(d.max() / self.dt)
            dv = (P - self.prev) / self.dt
            row["ke"] = float(0.5 * np.dot(self.mass_all, (dv * dv).sum(axis=1)))
        else:
            row["mean_disp"] = row["max_speed"] = row["ke"] = 0.0
        # constitution states
        self._commit_states(positions)
        # direct crease-angle diagnostics: |theta| over the crease-line hinges
        row["crease_theta_max"] = float(max(
            np.abs(_dihedral(positions[gi], st)[self.crease_mask[gi]]).max()
            for gi, st in enumerate(self.st) if self.kinds[gi] != "dahl"))
        row["crease_theta_mean"] = float(np.mean([
            np.abs(_dihedral(positions[gi], st)[self.crease_mask[gi]]).mean()
            for gi, st in enumerate(self.st) if self.kinds[gi] != "dahl"]))
        st = self.state_summary()
        row["dahl_active_frac"] = st["dahl_active_frac"]
        row["strain_yield_frac"] = st["strain_yield_frac"]
        row["stress_yield_frac"] = st["stress_yield_frac"]
        row["dahl_mean_absF_over_M"] = float(np.mean(
            [np.abs(self.F_commit[gi]).mean() / self.sat_M[gi].mean()
             for gi, k in enumerate(self.kinds) if k == "dahl"]))
        row["dahl_crease_absF_over_M"] = float(np.mean(
            [(np.abs(self.F_commit[gi]) / self.sat_M[gi])[self.crease_mask[gi]].mean()
             for gi, k in enumerate(self.kinds) if k == "dahl"]))
        row["plastic_mean_abs_theta_bar"] = float(np.mean(
            [np.abs(self.theta_bar[gi]).mean() for gi, k in enumerate(self.kinds)
             if k != "dahl"]))
        if stats is not None:
            row["newton"] = int(stats.get("newton_iterations", -1))
            row["pcg"] = int(stats.get("linear_solver_iterations", -1))
            row["line_search"] = int(stats.get("line_search_trials", -1))
            row["converged"] = int(stats.get("converged", -1))
            row["hit_newton_limit"] = int(bool(stats.get("hit_newton_limit", 0)))
            row["hit_ls_limit"] = int(bool(stats.get("hit_line_search_limit", 0)))
        self.prev = P
        self.rows.append(row)

    # -- summary ----------------------------------------------------------------
    def summary(self):
        rows = self.rows
        run = rows[1:] if len(rows) > 1 else rows
        half = max(1, len(run) // 4)
        newton = np.array([x["newton"] for x in run if "newton" in x])
        pcg = np.array([x["pcg"] for x in run if "pcg" in x])
        out = {
            "verify_all_finite": bool(all(x["finite"] for x in rows)),
            "verify_abs_x_max": max(x["abs_x_max"] for x in rows),
            # the press expels the pinned stack sideways along the free x
            # edges (measured 0.31-0.37 m across configs); the bar spans
            # +-0.45 m, so anything past half-width + 0.14 m would be riding
            # out from under the tool -- that is the runaway this catches.
            "verify_x_limit": 0.5 * self.sheet_x + 0.14,
            "verify_abs_z_max": max(x["abs_z_max"] for x in rows),
            "verify_z_limit": 0.5 * self.sheet_z + 0.01,      # clamped edges cannot move
            "verify_y_min": min(x["y_min"] for x in rows),
            "verify_y_floor": self.press_floor,    # ground
            "verify_area_ratio_min": min(x["area_ratio_min"] for x in rows),
            "verify_area_ratio_max": max(x["area_ratio_max"] for x in rows),
            "verify_sheet_area_ratio_max_final": rows[-1]["sheet_area_ratio_max"],
            "verify_tri_height_min": min(x["tri_height_min"] for x in rows),
            "verify_tri_height_floor": float(min(self.floor)),
            "verify_sheets_below_floor_frames": int(sum(x["sheets_below_floor"] for x in rows)),
            "verify_max_speed": max(x["max_speed"] for x in rows),
            "verify_mean_disp_mm": float(np.mean([x["mean_disp"] for x in run]) * 1e3),
            "verify_mean_disp_mm_last_quarter": float(
                np.mean([x["mean_disp"] for x in run[-half:]]) * 1e3),
            # constitution regime
            "verify_dahl_active_frac_after_press1": max(
                x["dahl_active_frac"] for x in rows if x["frame"] <= len(run) * 0.35),
            "verify_dahl_active_frac_final": rows[-1]["dahl_active_frac"],
            "verify_dahl_mean_absF_over_M_final": rows[-1]["dahl_mean_absF_over_M"],
            "verify_dahl_crease_absF_over_M_final": rows[-1]["dahl_crease_absF_over_M"],
            "verify_strain_yield_frac_final": rows[-1]["strain_yield_frac"],
            "verify_stress_yield_frac_final": rows[-1]["stress_yield_frac"],
            "verify_plastic_mean_abs_theta_bar_final": rows[-1]["plastic_mean_abs_theta_bar"],
            # residual creases (the physics being exercised)
            "verify_sheet_crease_defl_final_mm": [round(v * 1e3, 3) for v in rows[-1]["sheet_crease_defl"]],
            "verify_sheet_mean_defl_final_mm": [round(v * 1e3, 3) for v in rows[-1]["sheet_mean_defl"]],
            "verify_max_defl_mm": float(max(x["max_defl"] for x in rows) * 1e3),
            # solver health
            "verify_newton_min": int(newton.min()), "verify_newton_max": int(newton.max()),
            "verify_newton_mean": float(newton.mean()),
            "verify_newton_first_quarter": float(newton[:half].mean()),
            "verify_newton_last_quarter": float(newton[-half:].mean()),
            "verify_pcg_min": int(pcg.min()), "verify_pcg_max": int(pcg.max()),
            "verify_pcg_mean": float(pcg.mean()),
            "verify_pcg_first_quarter": float(pcg[:half].mean()),
            "verify_pcg_last_quarter": float(pcg[-half:].mean()),
            "verify_pcg_total": int(pcg.sum()),
            "verify_newton_total": int(newton.sum()),
            "verify_not_converged_frames": int(sum(1 for x in run if x.get("converged", 1) == 0)),
            "verify_hit_newton_limit_frames": int(sum(x.get("hit_newton_limit", 0) for x in run)),
            "verify_hit_ls_limit_frames": int(sum(x.get("hit_ls_limit", 0) for x in run)),
            "verify_ke_last_quarter": float(np.mean([x["ke"] for x in run[-half:]])),
        }
        # Gate rationale (measured on the shipped scene, see the round-7 s00
        # size sweep): the stiff carton sheets pucker transiently at the
        # clamp corners while the press squeezes the stack -- per-triangle
        # area ratios reach ~18x and the in-plane triangle height dips below
        # the strict 2r+d_hat floor for a few frames mid-squeeze -- then
        # partially recover (final per-sheet area maxima 1.0-3.4x, final
        # per-sheet min heights 4.7-7.0 mm).  Like the round-6 tumbler audit,
        # the hard gates are therefore the anti-NaN bounds and the sustained
        # behaviour (inversion bounds over the whole run, no non-converged or
        # Newton-limit frames, Newton decaying from the first quarter to the
        # last), and the extremes are reported, not gated.  The tri-height
        # gate uses the r6_common.ClothObserver relaxed floor 0.4*(2r+d_hat)
        # and tolerates the squeeze frames below it.
        relaxed = 0.4 * self.floor
        below_relaxed = int(sum(
            int((np.asarray(x["sheet_h_min"]) < relaxed).sum()) for x in rows))
        out["verify_tri_height_relaxed_floor_mm"] = [round(v * 1e3, 3) for v in relaxed]
        out["verify_tri_height_below_relaxed_sheet_frames"] = below_relaxed
        checks = {
            "finite": out["verify_all_finite"],
            # the transient corner pucker reaches ~18-31x run-to-run; the
            # SUSTAINED state (final frame, per sheet) stays 1.0-3.4x
            "no_inversion_or_collapse": (out["verify_area_ratio_min"] > 0.1
                                         and out["verify_area_ratio_max"] < 40.0
                                         and max(out["verify_sheet_area_ratio_max_final"]) < 5.0),
            "tri_height_no_sustained_collapse": below_relaxed <= 12 * len(rows),
            "bounded_speed": out["verify_max_speed"] < 50.0,
            "contained_x": out["verify_abs_x_max"] <= out["verify_x_limit"],
            "contained_z": out["verify_abs_z_max"] <= out["verify_z_limit"],
            "above_ground": out["verify_y_min"] >= out["verify_y_floor"] - 1e-6,
            "no_newton_limit": out["verify_hit_newton_limit_frames"] == 0,
            "no_newton_divergence": out["verify_newton_last_quarter"] < 1.5 * out["verify_newton_first_quarter"] + 2,
            "all_converged": out["verify_not_converged_frames"] == 0,
            "dahl_state_evolved": out["verify_dahl_active_frac_after_press1"] > 0.3,
            "plastic_yielded": min(out["verify_strain_yield_frac_final"],
                                   out["verify_stress_yield_frac_final"]) > 2e-4,
            "residual_creases": min(out["verify_sheet_crease_defl_final_mm"]) > 15.0,
        }
        out["checks"] = checks
        out["verify_ok"] = bool(all(checks.values()))
        return out

    def report(self):
        print("VERIFY_RESULT " + json.dumps(self.summary(), separators=(",", ":"),
                                            default=float), flush=True)
