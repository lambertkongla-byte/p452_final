"""
Coordinate-based tilted dipolar ladder simulation using QuSpin.

This module implements the dipolar zigzag chain quantum many-body scar
from "Exact quantum scars from kinetic frustration for cross-platform
realizations" (arXiv:2603.11191).

The chain is a 2xL "twisted ladder" of dipolar spins-1/2 interacting via the
spin-exchange Hamiltonian
    H = sum_{i<j} (J_ij/2) (S+_i S-_j + S-_i S+_j),
with couplings J_ij = (1 - 3 cos^2 theta_ij) / r_ij^3 given by the dipolar
form. By tuning the zigzag angles (alpha, beta) and the quantization-axis
direction, the diagonal couplings J_02 and J_13 become equal in magnitude
but opposite in sign, while the plaquette-diagonal couplings J_03 and J_12
are suppressed at the magic angle. The product state |d-, d-, ...> (or
equivalently |01 01 ...> in computational basis) is then a kinetic-frustration
scar.

Public API
----------
- build_all_dipolar_couplings_from_geometry
- build_all_dipolar_couplings_with_fixed_scale
- keep_designed_bonds_only
- coupling_table_rows
- get_bond, bond_category
- build_quspin_xy_hamiltonian
- sigma_z_operator, product_state_from_string
- evolve_state, compute_generalized_imbalance
- simulate_ladder         (high-level driver)
- add_z_position_noise    (disorder helper)
- plot_normalized_couplings, plot_pi_phase_lattice_mapping
- plot_quantization_axis_vector, set_pi_ticks
- make_imbalance_fidelity_figure
"""

import importlib
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

FIG_DPI = 170

# Make sure `geometry.py` is importable when this file is loaded from
# different working directories (e.g. via a Streamlit app one level up).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import geometry  # noqa: E402
geometry = importlib.reload(geometry)


# ============================================================================
# Coupling helpers
# ============================================================================

def get_bond(Jij, i, j):
    """Return J_ij from a Jij dictionary, agnostic to (i,j) ordering."""
    if i > j:
        i, j = j, i
    return Jij.get((i, j), 0.0)


def bond_category(i, j, L):
    """
    Categorize a bond on a 2xL ladder.

    Site convention:
        top:     0 -- 2 -- 4 -- ...
        bottom:  1 -- 3 -- 5 -- ...

    Categories: 'rung', 'top_leg', 'bottom_leg', 'plaquette_diagonal',
    'other_long_range'.
    """
    if i > j:
        i, j = j, i

    for r in range(L):
        if (i, j) == tuple(sorted((2 * r, 2 * r + 1))):
            return "rung"

    for r in range(L - 1):
        top_left = 2 * r
        bot_left = 2 * r + 1
        top_right = 2 * (r + 1)
        bot_right = 2 * (r + 1) + 1

        if (i, j) == tuple(sorted((top_left, top_right))):
            return "top_leg"
        if (i, j) == tuple(sorted((bot_left, bot_right))):
            return "bottom_leg"
        if (i, j) == tuple(sorted((top_left, bot_right))):
            return "plaquette_diagonal"
        if (i, j) == tuple(sorted((bot_left, top_right))):
            return "plaquette_diagonal"

    return "other_long_range"


def build_all_dipolar_couplings_from_geometry(
    coords,
    quantization_axis,
    normalize_bond=(0, 2),
    target_ref=1.0,
    max_distance=None,
):
    """
    Build all-pair dipolar couplings using geometry.compute_dipole_interaction.

    Raw coupling:
        J_ij_raw = [3 (qhat . rhat)^2 - 1] / r_ij^3

    All couplings are then rescaled so that the reference bond
    J_{normalize_bond} equals `target_ref` (typically +/-1).

    Returns
    -------
    Jij_norm : dict
        Normalized couplings, keyed by tuples (i, j) with i < j.
    Jij_raw : dict
        Un-normalized couplings, same keys.
    """
    coords = np.array(coords, dtype=float)
    N = len(coords)

    Jij_raw = {}
    for i in range(N):
        for j in range(i + 1, N):
            dist = geometry.compute_distances(coords[i], coords[j])
            if max_distance is not None and dist > max_distance:
                continue
            Jij_raw[(i, j)] = geometry.compute_dipole_interaction(
                coords[i], coords[j], quantization_axis,
            )

    i0, j0 = normalize_bond
    if i0 > j0:
        i0, j0 = j0, i0
    J_ref = Jij_raw[(i0, j0)]
    if abs(J_ref) < 1e-12:
        raise ValueError(
            f"Reference coupling J_{i0}{j0} is too close to zero to normalize by."
        )

    scale = target_ref / J_ref
    Jij_norm = {bond: scale * val for bond, val in Jij_raw.items()}
    return Jij_norm, Jij_raw


def build_all_dipolar_couplings_with_fixed_scale(
    coords,
    quantization_axis,
    scale,
    max_distance=None,
):
    """
    Build all-pair dipolar couplings using an externally fixed scale factor.

    Useful for noise simulations where the clean geometry sets the energy
    unit and noisy realizations must use the *same* scale so that J_02 is
    allowed to fluctuate.
    """
    coords = np.array(coords, dtype=float)
    N = len(coords)

    Jij = {}
    for i in range(N):
        for j in range(i + 1, N):
            dist = geometry.compute_distances(coords[i], coords[j])
            if max_distance is not None and dist > max_distance:
                continue
            Jij[(i, j)] = scale * geometry.compute_dipole_interaction(
                coords[i], coords[j], quantization_axis,
            )
    return Jij


def keep_designed_bonds_only(Jij, L):
    """
    Keep only nearest-neighbor bonds of the abstract 2xL ladder: rungs,
    top-leg bonds, and bottom-leg bonds.

    These are displayed in the UI as "only nearest neighbor interaction"
    because they are precisely the nearest-neighbor bonds in the effective
    ladder graph. In the paper's mapping, the kinetic-frustration bonds J_02
    and J_13 live on the ladder legs and have opposite signs, i.e. a pi-phase
    pattern.
    """
    keep = set()
    for r in range(L):
        keep.add(tuple(sorted((2 * r, 2 * r + 1))))
    for r in range(L - 1):
        keep.add(tuple(sorted((2 * r, 2 * (r + 1)))))
        keep.add(tuple(sorted((2 * r + 1, 2 * (r + 1) + 1))))

    return {bond: val for bond, val in Jij.items() if bond in keep}


def add_z_position_noise(coords, sigma_z, rng):
    """
    Add iid Gaussian z-noise to every site. `sigma_z` is in the same length
    units as `coords` (so with dx=1, sigma_z=0.01 means 1% of the rung
    spacing).
    """
    coords_noisy = np.array(coords, dtype=float).copy()
    coords_noisy[:, 2] += rng.normal(loc=0.0, scale=sigma_z, size=len(coords_noisy))
    return coords_noisy


# ============================================================================
# QuSpin helpers
# ============================================================================

def build_quspin_xy_hamiltonian(basis, Jij):
    """
    Build the XY spin-exchange Hamiltonian
        H = sum_{i<j} J_ij (Sx_i Sx_j + Sy_i Sy_j)
    in the supplied basis. With pauli=False, QuSpin uses Sx, Sy, Sz, so this
    matches the H_SE = sum (J/2) (S+S- + S-S+) form of the paper.
    """
    J_list = [
        [val, i, j]
        for (i, j), val in Jij.items()
        if abs(val) > 1e-14
    ]
    static = [["xx", J_list], ["yy", J_list]]

    H = hamiltonian(
        static, [], basis=basis, dtype=np.float64,
        check_herm=False, check_symm=False, check_pcon=False,
    )
    return H


def sigma_z_operator(basis, site):
    """
    Return Pauli sigma_z on `site` (with pauli=False, QuSpin's 'z' is Sz,
    so we multiply by 2 to get sigma_z with eigenvalues +/-1).
    """
    return hamiltonian(
        [["z", [[2.0, site]]]], [], basis=basis, dtype=np.float64,
        check_herm=False, check_symm=False, check_pcon=False,
    )


def product_state_from_string(basis, state_str):
    """Build a computational-basis product state from a QuSpin spin string."""
    psi = np.zeros(basis.Ns, dtype=np.complex128)
    psi[basis.index(state_str)] = 1.0
    return psi


def evolve_state(H, psi0, times):
    """
    Evolve psi0 under H. Returns array of shape (len(times), basis.Ns),
    regardless of which orientation QuSpin returns.
    """
    psis = H.evolve(psi0, 0.0, times)
    if psis.shape[0] == psi0.size:
        psis = psis.T
    return psis


def compute_generalized_imbalance(z_ops, psis_t, sigma0):
    """
    Generalized imbalance:
        I(t) = (1/N) sum_i <sigma_z_i(t)> sigma_z_i(0)
    """
    N = len(sigma0)
    out = np.zeros(psis_t.shape[0], dtype=float)
    for ti, psi in enumerate(psis_t):
        val = 0.0
        for i in range(N):
            zi = np.vdot(psi, z_ops[i].dot(psi)).real
            val += zi * sigma0[i]
        out[ti] = val / N
    return out


def _run_one_hamiltonian(basis, psi0, z_ops, sigma0, Jij, times):
    """Evolve under Jij and return (site_mag, imbalance, fidelity)."""
    H = build_quspin_xy_hamiltonian(basis, Jij)
    psis_t = evolve_state(H, psi0, times)

    N = len(sigma0)
    site_mag = np.zeros((psis_t.shape[0], N))
    for ti, psi in enumerate(psis_t):
        for i in range(N):
            site_mag[ti, i] = np.vdot(psi, z_ops[i].dot(psi)).real

    imbalance = (site_mag * sigma0[None, :]).mean(axis=1)
    fidelity = np.abs(psis_t @ psi0.conj()) ** 2
    return site_mag, imbalance, fidelity


# ============================================================================
# High-level driver
# ============================================================================

def simulate_ladder(
    L,
    alpha_deg=34.0,
    beta_deg=32.45,
    theta_deg=48.2,
    quantization_axis=(0.0, np.cos(np.pi / 6), np.sin(np.pi / 6)),
    n_pi=8.0,
    num_points=800,
    include_full=True,
    include_designed=True,
):
    """
    Simulate a 2xL tilted dipolar ladder, returning a dict of results.

    Parameters
    ----------
    L : int
        Number of pairs (rungs); total sites N = 2L.
    alpha_deg, beta_deg : float
        Zigzag angles (degrees) defining the chain geometry.
    theta_deg : float
        In-plane rotation angle passed to geometry.coords_for_zigzag_chain.
    quantization_axis : 3-tuple
        Direction of the dipolar quantization axis.
    n_pi : float
        Total simulation time in units of pi / |J_02|.
    num_points : int
        Number of time points.
    include_full / include_designed : bool
        Toggle the two model variants.

    Returns
    -------
    dict with keys:
        coords, Jij_full, Jij_designed,
        times, x_axis, J01, J02, J13, J03, J12,
        site_mag_full, imb_full, fid_full,           (if include_full)
        site_mag_designed, imb_designed, fid_designed (if include_designed),
        sigma0, sublattice, basis_dim, state_str.
    """
    N = 2 * L
    Nup = L

    coords = np.array(
        geometry.coords_for_zigzag_chain(
            n_pairs=L, theta_deg=theta_deg,
            alpha_deg=alpha_deg, beta_deg=beta_deg, dx=1.0,
        ),
        dtype=float,
    )

    # Normalize so that |J_02| = 1, preserving original signs.
    J02_raw = geometry.compute_dipole_interaction(
        coords[0], coords[2], quantization_axis,
    )
    if abs(J02_raw) < 1e-12:
        raise ValueError("Raw J_02 ~ 0; cannot normalize by it for this geometry.")
    target_J02 = float(np.sign(J02_raw))

    Jij_full, _Jij_raw = build_all_dipolar_couplings_from_geometry(
        coords=coords, quantization_axis=quantization_axis,
        normalize_bond=(0, 2), target_ref=target_J02,
    )
    Jij_designed = keep_designed_bonds_only(Jij_full, L)

    J02_abs = abs(get_bond(Jij_full, 0, 2))
    times = np.linspace(0.0, n_pi * np.pi / J02_abs, num_points)
    x_axis = J02_abs * times

    basis = spin_basis_1d(N, Nup=Nup, pauli=False)
    state_str = "01" * L  # site 0 down, site 1 up, ...
    psi0 = product_state_from_string(basis, state_str)
    sigma0 = np.array([+1 if c == "1" else -1 for c in state_str], dtype=float)
    z_ops = [sigma_z_operator(basis, i).tocsr() for i in range(N)]

    out = {
        "L": L, "N": N, "Nup": Nup,
        "coords": coords,
        "Jij_full": Jij_full,
        "Jij_designed": Jij_designed,
        "times": times, "x_axis": x_axis,
        "J01": get_bond(Jij_full, 0, 1),
        "J02": get_bond(Jij_full, 0, 2),
        "J13": get_bond(Jij_full, 1, 3),
        "J03": get_bond(Jij_full, 0, 3),
        "J12": get_bond(Jij_full, 1, 2),
        "sigma0": sigma0,
        "sublattice": sigma0,
        "state_str": state_str,
        "basis_dim": basis.Ns,
        "quantization_axis": tuple(quantization_axis),
    }

    if include_designed:
        sm_d, imb_d, fid_d = _run_one_hamiltonian(
            basis, psi0, z_ops, sigma0, Jij_designed, times,
        )
        out["site_mag_designed"] = sm_d
        out["imb_designed"] = imb_d
        out["fid_designed"] = fid_d

    if include_full:
        sm_f, imb_f, fid_f = _run_one_hamiltonian(
            basis, psi0, z_ops, sigma0, Jij_full, times,
        )
        out["site_mag_full"] = sm_f
        out["imb_full"] = imb_f
        out["fid_full"] = fid_f

    return out


def simulate_noise_ensemble(
    L,
    sigma_z=0.02,
    n_shots=8,
    seed=1,
    alpha_deg=34.0,
    beta_deg=32.45,
    theta_deg=48.2,
    quantization_axis=(0.0, np.cos(np.pi / 6), np.sin(np.pi / 6)),
    n_pi=8.0,
    num_points=600,
):
    """
    Simulate a 2xL ladder with z-position disorder, averaging over n_shots.

    Returns a dict with x_axis, individual shots, mean and std for both
    imbalance and fidelity, plus the clean reference.
    """
    rng = np.random.default_rng(seed)

    # Clean geometry sets the energy scale.
    coords_clean = np.array(
        geometry.coords_for_zigzag_chain(
            n_pairs=L, theta_deg=theta_deg,
            alpha_deg=alpha_deg, beta_deg=beta_deg, dx=1.0,
        ),
        dtype=float,
    )
    J02_raw_clean = geometry.compute_dipole_interaction(
        coords_clean[0], coords_clean[2], quantization_axis,
    )
    target_J02 = float(np.sign(J02_raw_clean))
    global_scale = target_J02 / J02_raw_clean

    Jij_clean_full = build_all_dipolar_couplings_with_fixed_scale(
        coords=coords_clean, quantization_axis=quantization_axis,
        scale=global_scale,
    )
    J02_abs_clean = abs(get_bond(Jij_clean_full, 0, 2))

    N = 2 * L
    Nup = L
    times = np.linspace(0.0, n_pi * np.pi / J02_abs_clean, num_points)
    x_axis = J02_abs_clean * times

    basis = spin_basis_1d(N, Nup=Nup, pauli=False)
    state_str = "01" * L
    psi0 = product_state_from_string(basis, state_str)
    sigma0 = np.array([+1 if c == "1" else -1 for c in state_str], dtype=float)
    z_ops = [sigma_z_operator(basis, i).tocsr() for i in range(N)]

    # Clean reference.
    _, imb_clean, fid_clean = _run_one_hamiltonian(
        basis, psi0, z_ops, sigma0, Jij_clean_full, times,
    )

    imb_shots = np.zeros((n_shots, num_points))
    fid_shots = np.zeros((n_shots, num_points))

    for s in range(n_shots):
        coords_noisy = add_z_position_noise(coords_clean, sigma_z, rng)
        Jij_noisy = build_all_dipolar_couplings_with_fixed_scale(
            coords=coords_noisy, quantization_axis=quantization_axis,
            scale=global_scale,
        )
        _, imb_shots[s], fid_shots[s] = _run_one_hamiltonian(
            basis, psi0, z_ops, sigma0, Jij_noisy, times,
        )

    return {
        "L": L,
        "times": times,
        "x_axis": x_axis,
        "imb_clean": imb_clean,
        "fid_clean": fid_clean,
        "imb_shots": imb_shots,
        "fid_shots": fid_shots,
        "imb_mean": imb_shots.mean(axis=0),
        "imb_std": imb_shots.std(axis=0),
        "fid_mean": fid_shots.mean(axis=0),
        "fid_std": fid_shots.std(axis=0),
        "sigma_z": sigma_z,
        "n_shots": n_shots,
        "seed": seed,
        "n_pi": n_pi,
    }


# ============================================================================
# Plotting helpers
# ============================================================================

def set_pi_ticks(ax, xmax):
    """Set x-axis ticks at integer multiples of pi (dimensionless axis)."""
    nmax = int(np.floor(xmax / np.pi))
    ticks = [n * np.pi for n in range(nmax + 1)]
    labels = []
    for n in range(nmax + 1):
        if n == 0:
            labels.append(r"$0$")
        elif n == 1:
            labels.append(r"$\pi$")
        else:
            labels.append(rf"${n}\pi$")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _build_value_color_map(Jij, group_precision=3, cmap_name="tab10"):
    """{rounded_J -> color}, so bonds with equal J share a color."""
    groups = {}
    for (i, j), val in Jij.items():
        key = round(val, group_precision)
        groups.setdefault(key, []).append((i, j))
    sorted_keys = sorted(groups.keys(), key=lambda k: (-abs(k), -k))
    cmap = plt.get_cmap(cmap_name)
    color_for_key = {k: cmap(idx % cmap.N) for idx, k in enumerate(sorted_keys)}
    return color_for_key, sorted_keys, groups



def coupling_table_rows(coords, Jij, quantization_axis, L):
    """
    Return a sorted list of coupling-table rows for UI display or printing.

    The dipole angle reported here is the 3D angle between the quantization
    axis and the bond direction, folded to [0, 90] deg because the dipolar
    factor depends on cos^2(theta_ij).
    """
    coords = np.array(coords, dtype=float)
    q = np.array(quantization_axis, dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-14:
        raise ValueError("quantization_axis must be nonzero.")
    q = q / q_norm

    rows = []
    for (i, j), val in Jij.items():
        r_vec = coords[j] - coords[i]
        dist = float(np.linalg.norm(r_vec))
        if dist < 1e-14:
            theta_deg = float("nan")
        else:
            r_hat = r_vec / dist
            cos_theta = float(np.dot(q, r_hat))
            theta_deg = float(np.rad2deg(
                np.arccos(np.clip(abs(cos_theta), 0.0, 1.0))
            ))
        rows.append({
            "Bond": f"J_{i}{j}",
            "Sites": f"{i}-{j}",
            "Category": bond_category(i, j, L).replace("_", " "),
            "Distance": dist,
            "3D dipole angle θ_ij (deg)": theta_deg,
            "J_ij / |J_02|": float(val),
            "|J_ij| / |J_02|": float(abs(val)),
        })
    rows.sort(key=lambda r: -r["|J_ij| / |J_02|"])
    return rows


def plot_quantization_axis_vector(
    quantization_axis,
    coords=None,
    ax=None,
    title="Quantization axis in 3D",
    show_site_labels=False,
    elev=22,
    azim=-55,
):
    """
    Plot the 3D quantization-axis vector used in the dipolar interaction.

    If coords are provided, the current chain geometry is also projected into
    the same 3D axes, so the user can see whether qhat is in-plane or tilted
    out of the chain plane.
    """
    q = np.array(quantization_axis, dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-14:
        raise ValueError("quantization_axis must be nonzero.")
    q = q / q_norm

    if ax is None:
        fig = plt.figure(figsize=(6.8, 5.8), dpi=FIG_DPI)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        fig.set_dpi(FIG_DPI)

    axis_len = 1.0
    ax.quiver(0, 0, 0, axis_len, 0, 0, arrow_length_ratio=0.10,
              linewidth=1.0, color="0.55")
    ax.quiver(0, 0, 0, 0, axis_len, 0, arrow_length_ratio=0.10,
              linewidth=1.0, color="0.55")
    ax.quiver(0, 0, 0, 0, 0, axis_len, arrow_length_ratio=0.10,
              linewidth=1.0, color="0.55")
    ax.text(axis_len * 1.08, 0, 0, "x", color="0.35")
    ax.text(0, axis_len * 1.08, 0, "y", color="0.35")
    ax.text(0, 0, axis_len * 1.08, "z", color="0.35")

    q_color = "#16a34a"
    ax.quiver(0, 0, 0, q[0], q[1], q[2], arrow_length_ratio=0.14,
              linewidth=2.0, color=q_color)
    ax.text(q[0] * 1.10, q[1] * 1.10, q[2] * 1.10,
            "q-hat", color=q_color, fontweight="bold")

    if coords is not None:
        xyz = np.array(coords, dtype=float)
        site_colors = ["tab:red" if i % 2 == 0 else "tab:blue"
                       for i in range(len(xyz))]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=58,
                   color=site_colors, edgecolor="white", linewidth=0.8,
                   depthshade=False)
        if show_site_labels:
            for i, (x, y, z) in enumerate(xyz):
                spin = "↓" if i % 2 == 0 else "↑"
                ax.text(x, y, z, f" {i}{spin}", fontsize=9)
        for i in range(0, len(xyz) - 2, 2):
            ax.plot([xyz[i, 0], xyz[i + 2, 0]],
                    [xyz[i, 1], xyz[i + 2, 1]],
                    [xyz[i, 2], xyz[i + 2, 2]], color="0.35", lw=1.0)
            ax.plot([xyz[i + 1, 0], xyz[i + 3, 0]],
                    [xyz[i + 1, 1], xyz[i + 3, 1]],
                    [xyz[i + 1, 2], xyz[i + 3, 2]], color="0.35", lw=1.0)
        for i in range(0, len(xyz), 2):
            if i + 1 < len(xyz):
                ax.plot([xyz[i, 0], xyz[i + 1, 0]],
                        [xyz[i, 1], xyz[i + 1, 1]],
                        [xyz[i, 2], xyz[i + 1, 2]], color="0.55", lw=0.8)

    lim = 1.25
    if coords is not None:
        xyz = np.array(coords, dtype=float)
        max_span = max(1.0, np.max(np.abs(xyz)))
        lim = 1.10 * max_span
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    return fig


def _bond_phase_label(val):
    """Return a compact phase label for a signed effective hopping/coupling."""
    return "0" if val >= 0 else "π"


def plot_pi_phase_lattice_mapping(
    Jij,
    L,
    ax=None,
    title="Mapped π-phase ladder",
    label_digits=2,
    show_coupling_values=True,
    show_phase_labels=True,
):
    """
    Plot the abstract 2xL nearest-neighbor ladder with bond signs interpreted
    as a 0/π phase pattern.

    Positive J is labelled phase 0; negative J is labelled phase π. This is
    the clean nearest-neighbor graph that the tilted all-pair dipolar chain is
    designed to approximate.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.6, 4.1), dpi=FIG_DPI)
    else:
        fig = ax.figure
        fig.set_dpi(FIG_DPI)

    pos = {}
    for r in range(L):
        pos[2 * r] = (float(r), 0.50)
        pos[2 * r + 1] = (float(r), -0.50)

    def draw_bond(i, j, width=2.6):
        val = get_bond(Jij, i, j)
        xi, yi = pos[i]
        xj, yj = pos[j]
        ls = "-" if val >= 0 else "--"
        color = "C0" if val >= 0 else "C3"
        ax.plot([xi, xj], [yi, yj], ls=ls, lw=width, color=color, alpha=0.92)
        xm, ym = 0.5 * (xi + xj), 0.5 * (yi + yj)
        phase = _bond_phase_label(val)
        if show_phase_labels or show_coupling_values:
            label = phase if show_phase_labels else ""
            if show_coupling_values:
                label = f"{phase}\n{val:+.{label_digits}f}"
            ax.text(
                xm, ym + (0.12 if abs(yi - yj) < 1e-8 else 0.0),
                label,
                ha="center", va="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                          lw=1.0, alpha=0.95),
            )

    for r in range(L):
        draw_bond(2 * r, 2 * r + 1, width=1.8)
    for r in range(L - 1):
        draw_bond(2 * r, 2 * (r + 1), width=3.0)
        draw_bond(2 * r + 1, 2 * (r + 1) + 1, width=3.0)

    for i, (x, y) in pos.items():
        ax.scatter([x], [y], s=180, color="k", edgecolor="white",
                   linewidth=1.2, zorder=5)
        ax.text(x, y, str(i), color="white", ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=6)

    ax.text(-0.35, 0.50, "top", ha="right", va="center", fontsize=10)
    ax.text(-0.35, -0.50, "bottom", ha="right", va="center", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.65, max(1.0, L - 1) + 0.55)
    ax.set_ylim(-0.95, 0.95)
    ax.set_axis_off()
    ax.set_title(title)
    ax.text(
        0.01, 0.03,
        "solid = positive phase 0, dashed = negative phase π",
        transform=ax.transAxes, ha="left", va="top", fontsize=9, color="0.35",
    )
    return fig

def plot_normalized_couplings(
    coords, Jij, L,
    ax=None,
    title="Normalized couplings used in QuSpin",
    label_digits=2,
    label_scale=0.20,
    group_precision=3,
    show_legend=True,
    show_bond_labels=True,
    show_site_labels=True,
    site_label_mode="index",
    use_coupling_colormap=False,
    show_colorbar=False,
    cmap_name="RdBu_r",
):
    """
    Plot normalized QuSpin couplings on a 2D projection.

    If `ax` is None a new figure is created. Returns the figure.
    """
    coords = np.array(coords, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=FIG_DPI)
    else:
        fig = ax.figure
        fig.set_dpi(FIG_DPI)

    if len(Jij) == 0:
        ax.set_title(title + " (empty)")
        return fig

    max_abs_J = max(abs(v) for v in Jij.values())
    if use_coupling_colormap:
        cmap = plt.get_cmap(cmap_name)
        if any(v < 0 for v in Jij.values()) and any(v > 0 for v in Jij.values()):
            norm = TwoSlopeNorm(vmin=-max_abs_J, vcenter=0.0, vmax=max_abs_J)
        else:
            vmin = min(Jij.values())
            vmax = max(Jij.values())
            if abs(vmax - vmin) < 1e-12:
                vmin -= 1.0
                vmax += 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
        color_for_key, sorted_keys = {}, []
    else:
        color_for_key, sorted_keys, _ = _build_value_color_map(
            Jij, group_precision=group_precision,
        )

    for idx, ((i, j), val) in enumerate(sorted(Jij.items())):
        xi, yi = coords[i, 0], coords[i, 1]
        xj, yj = coords[j, 0], coords[j, 1]
        key = round(val, group_precision)
        color = cmap(norm(val)) if use_coupling_colormap else color_for_key[key]

        lw = 1.0 + 2.5 * abs(val) / max_abs_J
        alpha = 0.55 + 0.4 * abs(val) / max_abs_J
        category = bond_category(i, j, L)
        linestyle = "--" if category not in {"rung", "top_leg", "bottom_leg"} else "-"
        ax.plot(
            [xi, xj], [yi, yj],
            color=color, linewidth=lw, alpha=alpha, linestyle=linestyle, zorder=1,
        )

        xm = 0.5 * (xi + xj)
        ym = 0.5 * (yi + yj)
        bond_vec = np.array([xj - xi, yj - yi])
        bond_norm = np.linalg.norm(bond_vec)
        if bond_norm > 1e-12:
            bond_unit = bond_vec / bond_norm
            perp = np.array([-bond_unit[1], bond_unit[0]])
        else:
            perp = np.array([0.0, 0.0])

        if category == "top_leg":
            sign = +1.0
        elif category == "bottom_leg":
            sign = -1.0
        elif category == "plaquette_diagonal":
            sign = +1.0 if (i % 2 == 0) else -1.0
        elif category == "rung":
            sign = +0.6
        else:
            sign = +1.0 if (idx % 2 == 0) else -1.0
        offset = sign * label_scale * perp

        if show_bond_labels:
            ax.text(
                xm + offset[0], ym + offset[1],
                f"{val:+.{label_digits}f}",
                fontsize=10, fontweight="bold", color=color,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec=color, lw=1.2, alpha=0.95),
                zorder=5,
            )

    if site_label_mode == "spin":
        site_colors = ["tab:red" if i % 2 == 0 else "tab:blue"
                       for i in range(len(coords))]
    else:
        site_colors = "black"
    ax.scatter(
        coords[:, 0], coords[:, 1], s=130,
        color=site_colors, edgecolor="white", linewidth=1.4, zorder=3,
    )
    if show_site_labels:
        for i, (x, y, _z) in enumerate(coords):
            if site_label_mode == "spin":
                label = f"{i}↓" if i % 2 == 0 else f"{i}↑"
                ax.text(x, y + 0.09, label, fontsize=10,
                        fontweight="bold", ha="center",
                        color="black", zorder=6)
            else:
                ax.text(x + 0.04, y + 0.04, str(i), fontsize=12,
                        fontweight="bold", color="black", zorder=6)

    if show_colorbar and use_coupling_colormap:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label(r"$J_{ij}/|J_{02}|$", rotation=90)

    if show_legend and not use_coupling_colormap:
        legend_handles = [
            Line2D([0], [0], color=color_for_key[k], lw=3,
                   label=f"J = {k:+.{label_digits + 1}f}")
            for k in sorted_keys
        ]
        ax.legend(
            handles=legend_handles, loc="center left",
            bbox_to_anchor=(1.02, 0.5), fontsize=8,
            title="Coupling values", title_fontsize=9, frameon=True,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.margins(0.12)
    for spine in ax.spines.values():
        spine.set_color("0.80")
    return fig


def make_imbalance_fidelity_figure(
    result, show_designed=True, show_full=True, cut_x=None,
):
    """
    Build a two-panel figure: imbalance (top) and fidelity (bottom),
    plotted against |J_02| t with pi-spaced ticks.
    """
    x = result["x_axis"]

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4),
                             sharex=True, dpi=FIG_DPI)

    # --- imbalance ---
    if show_designed and "imb_designed" in result:
        axes[0].plot(x, result["imb_designed"], label="only nearest neighbor interaction", lw=1.8)
    if show_full and "imb_full" in result:
        axes[0].plot(x, result["imb_full"], "--",
                     label="all-pair dipolar interaction", lw=1.8)
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].set_ylabel(r"$I(t)$")
    axes[0].set_title("Generalized imbalance")
    axes[0].legend(fontsize=9)
    if cut_x is not None:
        axes[0].axvline(cut_x, color="0.15", lw=1.1, ls=":", alpha=0.85)

    # --- fidelity ---
    if show_designed and "fid_designed" in result:
        axes[1].plot(x, result["fid_designed"], label="only nearest neighbor interaction", lw=1.8)
    if show_full and "fid_full" in result:
        axes[1].plot(x, result["fid_full"], "--",
                     label="all-pair dipolar interaction", lw=1.8)
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_ylabel(r"$F(t) = |\langle\psi_0|\psi(t)\rangle|^2$")
    axes[1].set_xlabel(r"$|J_{02}|\,t$")
    axes[1].set_title("Many-body fidelity")
    axes[1].legend(fontsize=9)
    if cut_x is not None:
        axes[1].axvline(cut_x, color="0.15", lw=1.1, ls=":", alpha=0.85)

    set_pi_ticks(axes[1], xmax=x[-1])

    fig.tight_layout()
    return fig


def make_spin_configuration_figure(
    coords, site_mag, title="Spin configuration", ax=None,
):
    """Plot site-resolved <sigma_z> on the current ladder geometry."""
    coords = np.array(coords, dtype=float)
    site_mag = np.array(site_mag, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=FIG_DPI)
    else:
        fig = ax.figure
        fig.set_dpi(FIG_DPI)

    for i in range(0, len(coords) - 2, 2):
        ax.plot([coords[i, 0], coords[i + 2, 0]],
                [coords[i, 1], coords[i + 2, 1]], color="0.55", lw=1.4)
        ax.plot([coords[i + 1, 0], coords[i + 3, 0]],
                [coords[i + 1, 1], coords[i + 3, 1]], color="0.55", lw=1.4)
    for i in range(0, len(coords), 2):
        if i + 1 < len(coords):
            ax.plot([coords[i, 0], coords[i + 1, 0]],
                    [coords[i, 1], coords[i + 1, 1]], color="0.70", lw=1.1)

    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=site_mag, s=260,
        cmap="RdBu_r", vmin=-1.0, vmax=1.0,
        edgecolor="white", linewidth=1.3, zorder=3,
    )
    for i, (x, y, _z) in enumerate(coords):
        ax.text(x, y, f"{site_mag[i]:+.2f}", ha="center", va="center",
                fontsize=8.5, color="black", fontweight="bold", zorder=4)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"$\langle\sigma_i^z\rangle$")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)
    ax.margins(0.16)
    return fig


def make_noise_figure(noise_result):
    """Two-panel figure for imbalance and fidelity under z-position disorder."""
    x = noise_result["x_axis"]
    n_shots = noise_result["n_shots"]
    sigma_z = noise_result["sigma_z"]

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4),
                             sharex=True, dpi=FIG_DPI)

    for s in range(n_shots):
        axes[0].plot(x, noise_result["imb_shots"][s], color="gray", alpha=0.20, lw=0.8)
        axes[1].plot(x, noise_result["fid_shots"][s], color="gray", alpha=0.20, lw=0.8)

    axes[0].fill_between(
        x,
        noise_result["imb_mean"] - noise_result["imb_std"],
        noise_result["imb_mean"] + noise_result["imb_std"],
        alpha=0.18, lw=0,
    )
    axes[0].plot(x, noise_result["imb_mean"], lw=2.2,
                 label=rf"noisy avg, $\sigma_z={sigma_z:.3f}$")
    axes[0].plot(x, noise_result["imb_clean"], "--", lw=2.0, label="clean")
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].set_ylabel(r"$I(t)$")
    axes[0].set_title("Imbalance with z-position disorder")
    axes[0].legend(fontsize=9)

    axes[1].fill_between(
        x,
        noise_result["fid_mean"] - noise_result["fid_std"],
        noise_result["fid_mean"] + noise_result["fid_std"],
        alpha=0.18, lw=0,
    )
    axes[1].plot(x, noise_result["fid_mean"], lw=2.2,
                 label=rf"noisy avg, $\sigma_z={sigma_z:.3f}$")
    axes[1].plot(x, noise_result["fid_clean"], "--", lw=2.0, label="clean")
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_ylabel(r"$F(t)$")
    axes[1].set_xlabel(r"$|J_{02}^{\rm clean}|\,t$")
    axes[1].set_title("Fidelity with z-position disorder")
    axes[1].legend(fontsize=9)

    set_pi_ticks(axes[1], xmax=x[-1])
    fig.tight_layout()
    return fig


# ============================================================================
# Diagnostics (kept as utilities; only called from __main__)
# ============================================================================

def print_magic_diagnostics(Jij):
    """First-plaquette diagnostics for the kinetic-frustration condition."""
    J01 = get_bond(Jij, 0, 1)
    J02 = get_bond(Jij, 0, 2)
    J13 = get_bond(Jij, 1, 3)
    J03 = get_bond(Jij, 0, 3)
    J12 = get_bond(Jij, 1, 2)
    print("\n=== Magic/frustration diagnostics ===")
    print(f"J_01 = {J01:+.6f}")
    print(f"J_02 = {J02:+.6f}")
    print(f"J_13 = {J13:+.6f}")
    print(f"J_02 + J_13 = {J02 + J13:+.6e}   (should be ~0)")
    print(f"J_03 = {J03:+.6e}                 (should be ~0)")
    print(f"J_12 = {J12:+.6e}                 (should be ~0)")


def print_coupling_table(coords, Jij, quantization_axis, L,
                          title="Coupling strengths"):
    """Compact table of every coupling, sorted by |J| descending."""
    rows = coupling_table_rows(coords, Jij, quantization_axis, L)

    width = 96
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"  {'Bond':<8}{'Sites':<8}{'Category':<22}{'Distance':>10}"
          f"{'theta_ij':>12}{'J/|J02|':>14}{'|J|/|J02|':>14}")
    print("  " + "-" * (width - 2))
    for r in rows:
        print(f"  {r['Bond']:<8}{r['Sites']:<8}{r['Category']:<22}"
              f"{r['Distance']:>10.4f}"
              f"{r['3D dipole angle θ_ij (deg)']:>12.3f}"
              f"{r['J_ij / |J_02|']:>+14.5f}"
              f"{r['|J_ij| / |J_02|']:>14.5f}")
    print("=" * width)
    return rows


# ============================================================================
# Main demo
# ============================================================================

if __name__ == "__main__":
    L = 3
    alpha_deg = 34.0
    beta_deg = 32.45
    theta_deg = 48.2
    quantization_axis = (0.0, np.cos(np.pi / 6), np.sin(np.pi / 6))

    result = simulate_ladder(
        L=L,
        alpha_deg=alpha_deg, beta_deg=beta_deg, theta_deg=theta_deg,
        quantization_axis=quantization_axis,
        n_pi=8.0, num_points=2000,
    )

    print(f"\nN = {result['N']},  Hilbert dim = {result['basis_dim']}")
    print(f"J_01 = {result['J01']:+.5f}")
    print(f"J_02 = {result['J02']:+.5f}")
    print(f"J_13 = {result['J13']:+.5f}")
    print(f"|J_02/J_01| = {abs(result['J02']/result['J01']):.5f}")
    print(f"(J_02 + J_13)/|J_01| = "
          f"{(result['J02']+result['J13'])/abs(result['J01']):+.3e}")

    print_magic_diagnostics(result["Jij_full"])
    print_coupling_table(
        result["coords"], result["Jij_designed"],
        quantization_axis, L,
        title="Only nearest neighbor interaction",
    )

    fig_g = plot_normalized_couplings(
        result["coords"], result["Jij_full"], L,
        title=f"2 x {L}: all-pair dipolar interaction couplings",
    )

    fig_d = make_imbalance_fidelity_figure(result)
    plt.show()
