"""
Streamlit app for the QuSpin dipolar zigzag chain (kinetic-frustration scar).

Structure follows the previous Streamlit `app.py` style: one sidebar page
selector, cached helper functions, and page-local controls. The dipolar ladder
page contains tabs for display/mapping, clean dynamics, and z-position disorder.
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

try:
    import pandas as pd
except Exception:  # pragma: no cover - Streamlit normally installs pandas.
    pd = None

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - Plotly is optional; Matplotlib fallback below.
    go = None

FIG_DPI = 170
plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": 240,
})

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for path in (HERE, PARENT):
    if path not in sys.path:
        sys.path.insert(0, path)

from qspin_dipoles import (  # noqa: E402
    coupling_table_rows,
    get_bond,
    make_imbalance_fidelity_figure,
    make_noise_figure,
    make_spin_configuration_figure,
    plot_normalized_couplings,
    plot_pi_phase_lattice_mapping,
    plot_quantization_axis_vector,
    simulate_ladder,
    simulate_noise_ensemble,
)

MAGIC_DIPOLE_ANGLE_DEG = float(np.rad2deg(np.arccos(1.0 / np.sqrt(3.0))))


# =============================================================================
# Helpers
# =============================================================================

def safe_tight_layout(fig):
    """Use tight_layout when possible, but do not let Matplotlib crash Streamlit."""
    fig.set_dpi(FIG_DPI)
    try:
        fig.tight_layout()
    except Exception:
        fig.subplots_adjust(left=0.10, right=0.92, bottom=0.14, top=0.90,
                            wspace=0.32, hspace=0.32)


def make_interactive_quantization_axis_figure(
    quantization_axis, coords, view_elev_deg=24.0, view_azim_deg=-48.0,
):
    """Build a rotatable Plotly 3D qhat/geometry view."""
    if go is None:
        return None

    q = np.array(quantization_axis, dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-14:
        raise ValueError("quantization_axis must be nonzero.")
    q = q / q_norm

    axis_len = 1.0
    lim = 1.18
    q_end = q * axis_len
    q_color = "#16a34a"

    fig = go.Figure()

    def add_line(name, start, end, color, width=3, hover="skip"):
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo=hover,
            name=name,
            showlegend=False,
        ))

    def add_axis(name, end, color, width=2.2, head_size=0.075):
        shaft_end = 0.86 * end
        add_line(name, np.zeros(3), shaft_end, color, width=width)
        fig.add_trace(go.Cone(
            x=[shaft_end[0]], y=[shaft_end[1]], z=[shaft_end[2]],
            u=[end[0]], v=[end[1]], w=[end[2]],
            anchor="tail",
            sizemode="absolute",
            sizeref=head_size * axis_len,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            hoverinfo="skip",
            name=name,
            showlegend=False,
        ))

    add_axis("x", np.array([axis_len, 0.0, 0.0]), "#9ca3af", width=2)
    add_axis("y", np.array([0.0, axis_len, 0.0]), "#9ca3af", width=2)
    add_axis("z", np.array([0.0, 0.0, axis_len]), "#9ca3af", width=2)
    add_axis("q-hat", q_end, q_color, width=3, head_size=0.09)

    for text, end in (("x", [axis_len * 1.12, 0, 0]),
                      ("y", [0, axis_len * 1.12, 0]),
                      ("z", [0, 0, axis_len * 1.12])):
        fig.add_trace(go.Scatter3d(
            x=[end[0]], y=[end[1]], z=[end[2]],
            mode="text",
            text=[text],
            textfont=dict(color="#525252", size=13),
            hoverinfo="skip",
            showlegend=False,
        ))
    fig.add_trace(go.Scatter3d(
        x=[q_end[0] * 1.12], y=[q_end[1] * 1.12], z=[q_end[2] * 1.12],
        mode="text",
        text=["q-hat"],
        textfont=dict(color=q_color, size=15),
        hovertemplate=(
            "qhat = "
            f"({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f})"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    theta = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(go.Scatter3d(
        x=np.cos(theta), y=np.sin(theta), z=np.zeros_like(theta),
        mode="lines",
        line=dict(color="rgba(22, 163, 74, 0.18)", width=3),
        hoverinfo="skip",
        showlegend=False,
    ))

    elev = np.deg2rad(view_elev_deg)
    azim = np.deg2rad(view_azim_deg)
    camera_radius = 1.85
    camera_eye = dict(
        x=float(camera_radius * np.cos(elev) * np.cos(azim)),
        y=float(camera_radius * np.cos(elev) * np.sin(azim)),
        z=float(camera_radius * np.sin(elev)),
    )

    fig.update_layout(
        title="Quantization axis",
        height=430,
        margin=dict(l=0, r=0, t=42, b=0),
        dragmode="turntable",
        uirevision="q-axis",
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="x", range=[-lim, lim], showspikes=False),
            yaxis=dict(title="y", range=[-lim, lim], showspikes=False),
            zaxis=dict(title="z", range=[-lim, lim], showspikes=False),
            camera=dict(eye=camera_eye),
        ),
    )
    return fig


def magic_theta_from_quantization_axis(
    alpha_deg, beta_deg, quantization_axis, current_theta_deg=48.2,
):
    """Find in-plane theta where plaquette diagonals hit the dipolar magic angle."""
    import geometry as _geometry

    q = np.array(quantization_axis, dtype=float)
    q /= np.linalg.norm(q)
    target = 1.0 / np.sqrt(3.0)

    def objective(theta_deg):
        coords = np.array(
            _geometry.coords_for_zigzag_chain(
                n_pairs=2, theta_deg=float(theta_deg),
                alpha_deg=alpha_deg, beta_deg=beta_deg, dx=1.0,
            ),
            dtype=float,
        )
        errors = []
        for i, j in ((0, 3), (1, 2)):
            r = coords[j] - coords[i]
            r /= np.linalg.norm(r)
            errors.append(abs(abs(float(np.dot(q, r))) - target))
        return max(errors)

    def angle_error_deg(theta_deg):
        coords = np.array(
            _geometry.coords_for_zigzag_chain(
                n_pairs=2, theta_deg=float(theta_deg),
                alpha_deg=alpha_deg, beta_deg=beta_deg, dx=1.0,
            ),
            dtype=float,
        )
        errors = []
        for i, j in ((0, 3), (1, 2)):
            r = coords[j] - coords[i]
            r /= np.linalg.norm(r)
            angle = np.rad2deg(np.arccos(np.clip(abs(float(np.dot(q, r))), 0.0, 1.0)))
            errors.append(abs(angle - MAGIC_DIPOLE_ANGLE_DEG))
        return max(errors)

    grid = np.linspace(0.0, 180.0, 1801)
    errors = np.array([objective(theta) for theta in grid])
    min_error = float(np.min(errors))
    near_best = grid[errors <= min_error + 1e-5]
    best = float(near_best[np.argmin(np.abs(near_best - current_theta_deg))])
    fine = np.linspace(max(0.0, best - 0.3), min(180.0, best + 0.3), 1201)
    fine_errors = np.array([objective(theta) for theta in fine])
    min_fine_error = float(np.min(fine_errors))
    fine_best = fine[fine_errors <= min_fine_error + 1e-8]
    best = float(fine_best[np.argmin(np.abs(fine_best - current_theta_deg))])
    return best, float(angle_error_deg(best))


def selected_bond_angle_deg(coords, quantization_axis, i, j):
    """Folded 3D dipole angle for one bond."""
    q = np.array(quantization_axis, dtype=float)
    q /= np.linalg.norm(q)
    r = np.array(coords[j], dtype=float) - np.array(coords[i], dtype=float)
    r /= np.linalg.norm(r)
    return float(np.rad2deg(np.arccos(np.clip(abs(np.dot(q, r)), 0.0, 1.0))))


@st.cache_data(show_spinner=False)
def geometry_only(L, alpha_deg, beta_deg, theta_geo_deg, quantization_axis):
    """Compute coordinates and normalized all-pair dipolar couplings."""
    from qspin_dipoles import build_all_dipolar_couplings_from_geometry
    import geometry as _geometry

    coords = np.array(
        _geometry.coords_for_zigzag_chain(
            n_pairs=L,
            theta_deg=theta_geo_deg,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            dx=1.0,
        ),
        dtype=float,
    )

    J02_raw = _geometry.compute_dipole_interaction(
        coords[0], coords[2], quantization_axis,
    )
    if abs(J02_raw) < 1e-12:
        target_J02 = 1.0
    else:
        target_J02 = float(np.sign(J02_raw))

    Jij_full, _ = build_all_dipolar_couplings_from_geometry(
        coords=coords,
        quantization_axis=quantization_axis,
        normalize_bond=(0, 2),
        target_ref=target_J02,
    )
    return coords, Jij_full


def quantization_axis_from_angles(q_theta_deg, q_phi_deg):
    """Return qhat from 3D spherical angles: theta from z, phi from x in xy."""
    theta = np.deg2rad(q_theta_deg)
    phi = np.deg2rad(q_phi_deg)
    return (
        float(np.sin(theta) * np.cos(phi)),
        float(np.sin(theta) * np.sin(phi)),
        float(np.cos(theta)),
    )


def render_coupling_dataframe(coords, Jij_full, quantization_axis, L):
    """Render a clean coupling-strength table in Streamlit."""
    essential = {"J_01", "J_02", "J_13", "J_03", "J_12"}
    rows = [
        row for row in coupling_table_rows(coords, Jij_full, quantization_axis, L)
        if row["Bond"] in essential
    ]
    if pd is None:
        st.table(rows)
        return

    df = pd.DataFrame(rows)
    df = df[["Bond", "Category", "3D dipole angle θ_ij (deg)", "J_ij / |J_02|"]]
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "3D dipole angle θ_ij (deg)": st.column_config.NumberColumn(
                "3D dipole angle θ_ij (deg)", format="%.2f",
            ),
            "J_ij / |J_02|": st.column_config.NumberColumn(
                "J/|J_02|", format="%+.4f",
            ),
        },
    )


def render_kinetic_diagnostics(coords, Jij_full, quantization_axis, L,
                               N, hilbert_dim, J01, J02, J13, J03, J12):
    """Render metrics and coupling tables near the geometry view."""
    st.subheader("Kinetic-frustration diagnostics")
    theta_03 = selected_bond_angle_deg(coords, quantization_axis, 0, 3)
    theta_12 = selected_bond_angle_deg(coords, quantization_axis, 1, 2)
    theta_diag = 0.5 * (theta_03 + theta_12)
    theta_err = max(
        abs(theta_03 - MAGIC_DIPOLE_ANGLE_DEG),
        abs(theta_12 - MAGIC_DIPOLE_ANGLE_DEG),
    )
    magic_ok = theta_err <= 1.0
    leg_balance = abs(J02 + J13) / max(abs(J02), abs(J13), 1e-12)
    diag_leakage = max(abs(J03), abs(J12)) / max(abs(J02), 1e-12)
    kinetic_ok = leg_balance <= 0.05 and diag_leakage <= 0.10

    summary = st.columns(4)
    summary[0].metric("Sites N", N)
    summary[1].metric("Magic-angle check", "Yes" if magic_ok else "No",
                      f"diagonal angle {theta_diag:.2f}°")
    summary[2].metric("Kinetic frustration", "Yes" if kinetic_ok else "No",
                      f"leg mismatch {leg_balance:.2%}")
    summary[3].metric("Diagonal leakage", f"{diag_leakage:.2%}")

    diag_rows = [
        {"Check": "Magic angle on diagonals", "Value": f"{theta_diag:.2f}°",
         "Target": f"{MAGIC_DIPOLE_ANGLE_DEG:.2f}°", "Status": "OK" if magic_ok else "Tune θ"},
        {"Check": "J_02 + J_13 cancellation", "Value": f"{J02 + J13:+.3e}",
         "Target": "near 0", "Status": "OK" if leg_balance <= 0.05 else "Mismatch"},
        {"Check": "Plaquette diagonals", "Value": f"max={diag_leakage:.2%}",
         "Target": "< 10% of |J_02|", "Status": "OK" if diag_leakage <= 0.10 else "Large"},
        {"Check": "Sz sector dimension", "Value": f"C({N},{L}) = {hilbert_dim}",
         "Target": "simulation size", "Status": "Info"},
    ]
    if pd is not None:
        diag_df = pd.DataFrame(diag_rows)
        st.dataframe(diag_df, hide_index=True, use_container_width=True)
    else:
        st.table(diag_rows)

    st.markdown("#### Key coupling strengths")
    render_coupling_dataframe(coords, Jij_full, quantization_axis, L)
    with st.expander("Show all normalized couplings"):
        rows = coupling_table_rows(coords, Jij_full, quantization_axis, L)
        if pd is not None:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.table(rows)


def model_flags(interaction_choice):
    """Map the UI choice to simulate_ladder include flags and plot flags."""
    if interaction_choice == "only nearest neighbor interaction":
        return True, False
    if interaction_choice == "all-pair dipolar interaction":
        return False, True
    return True, True


# =============================================================================
# Streamlit page setup
# =============================================================================

st.set_page_config(
    page_title="Dipolar Scar - QuSpin",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Dipolar Scar")
page = st.sidebar.radio(
    "Section",
    ["Dipolar ladder demonstration", "About"],
)
st.sidebar.markdown("---")


# =============================================================================
# Main page
# =============================================================================

if page == "Dipolar ladder demonstration":
    st.title("Dipolar Zigzag Chain — QuSpin exact dynamics")
    st.markdown(
        "Exact-diagonalization simulation of the spin-exchange Hamiltonian "
        r"$H = \sum_{i<j} (J_{ij}/2)(S_i^+S_j^- + S_i^-S_j^+)$ "
        "on a 2×L twisted dipolar ladder. Site convention: "
        "**top leg = 0, 2, 4, ...** and **bottom leg = 1, 3, 5, ...**. "
        "The initial product state is "
        r"$|01\,01\,01\,\ldots\rangle$."
    )

    # -------------------------------------------------------------------------
    # Sidebar controls: only global geometry and quantization-axis parameters.
    # Run/noise controls are kept inside the relevant tabs.
    # -------------------------------------------------------------------------
    for key, value in {
        "alpha_deg": 34.0,
        "beta_deg": 32.45,
        "theta_inplane_deg": 48.2,
        "q_theta_deg": 60.0,
        "q_phi_deg": 90.0,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.sidebar.markdown("### Chain geometry")
    L = st.sidebar.slider("Number of spin pairs L (N = 2L sites)", 2, 6, 3)
    alpha_deg = st.sidebar.slider(
        "In-plane zigzag angle α (deg)", 20.0, 50.0, step=0.05,
        key="alpha_deg",
        help="In-plane angle used by geometry.coords_for_zigzag_chain.",
    )
    beta_deg = st.sidebar.slider(
        "In-plane zigzag angle β (deg)", 20.0, 50.0, step=0.05,
        key="beta_deg",
        help="In-plane angle used by geometry.coords_for_zigzag_chain.",
    )

    st.sidebar.markdown("### Quantization axis")
    q_theta_deg = st.sidebar.slider(
        "3D polar angle θ_q (deg)", 0.0, 180.0, step=1.0,
        key="q_theta_deg",
        help="3D spherical polar angle of qhat measured from the z-axis.",
    )
    q_phi_deg = st.sidebar.slider(
        "Azimuthal angle φ_q (deg)", 0.0, 360.0, step=1.0,
        key="q_phi_deg",
        help="Azimuth of qhat in the xy plane, measured from the x-axis.",
    )
    quantization_axis = quantization_axis_from_angles(q_theta_deg, q_phi_deg)
    if st.sidebar.button("Set θ to magic angle", use_container_width=True):
        theta_magic, magic_error = magic_theta_from_quantization_axis(
            alpha_deg, beta_deg, quantization_axis,
            current_theta_deg=st.session_state["theta_inplane_deg"],
        )
        st.session_state["theta_inplane_deg"] = round(theta_magic, 1)
        st.session_state["last_magic_error"] = magic_error

    theta_geo_deg = st.sidebar.slider(
        "In-plane θ rotation (deg)", 0.0, 180.0, step=0.1,
        key="theta_inplane_deg",
        help=(
            "Rotation of the planar ladder around the z-axis. The magic-angle "
            "button chooses the value that makes the plaquette diagonals "
            "closest to the dipolar magic angle for the current qhat."
        ),
    )
    st.sidebar.caption(
        f"qhat = ({quantization_axis[0]:+.3f}, "
        f"{quantization_axis[1]:+.3f}, {quantization_axis[2]:+.3f})"
    )
    if "last_magic_error" in st.session_state:
        st.sidebar.caption(
            f"Magic-angle residual: {st.session_state['last_magic_error']:.3f}°"
        )
    st.sidebar.info(
        "The ladder geometry is planar. θ is the in-plane rotation; θ_q and "
        "φ_q define the 3D dipole quantization axis."
    )

    coords, Jij_full = geometry_only(
        L, alpha_deg, beta_deg, theta_geo_deg, quantization_axis,
    )

    J01 = get_bond(Jij_full, 0, 1)
    J02 = get_bond(Jij_full, 0, 2)
    J13 = get_bond(Jij_full, 1, 3)
    J03 = get_bond(Jij_full, 0, 3)
    J12 = get_bond(Jij_full, 1, 2)
    N = 2 * L
    hilbert_dim = math.comb(N, L)

    display_tab, dynamics_tab, disorder_tab = st.tabs([
        "Display / mapping",
        "Imbalance and fidelity",
        "z-position disorder ensemble",
    ])

    # -------------------------------------------------------------------------
    # Display page with nested sub-tabs.
    # -------------------------------------------------------------------------
    with display_tab:
        geom_tab, pi_map_tab = st.tabs([
            "Geometry + q-axis",
            "π-phase lattice mapping",
        ])

        with geom_tab:
            col_geo, col_axis = st.columns([1.35, 1.0])
            with col_geo:
                st.subheader("Projected ladder geometry and normalized couplings")
                fig_geo, ax_geo = plt.subplots(figsize=(8.6, 5.2), dpi=FIG_DPI)
                plot_normalized_couplings(
                    coords,
                    Jij_full,
                    L,
                    ax=ax_geo,
                    title=f"2 × {L} tilted dipolar ladder",
                    label_digits=2,
                    label_scale=0.20,
                    show_legend=False,
                    show_bond_labels=False,
                    site_label_mode="spin",
                    use_coupling_colormap=True,
                    show_colorbar=True,
                )
                safe_tight_layout(fig_geo)
                st.pyplot(fig_geo)
                plt.close(fig_geo)

            with col_axis:
                st.subheader("Quantization-axis vector")
                view_cols = st.columns(2)
                with view_cols[0]:
                    q_view_azim = st.slider(
                        "View azimuth", -180, 180, -48, step=3,
                        key="q_view_azim",
                    )
                with view_cols[1]:
                    q_view_elev = st.slider(
                        "View elevation", -80, 80, 24, step=2,
                        key="q_view_elev",
                    )
                fig_q_interactive = make_interactive_quantization_axis_figure(
                    quantization_axis, coords,
                    view_elev_deg=q_view_elev,
                    view_azim_deg=q_view_azim,
                )
                if fig_q_interactive is not None:
                    st.plotly_chart(
                        fig_q_interactive,
                        use_container_width=True,
                        config={
                            "displaylogo": False,
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "responsive": True,
                        },
                    )
                else:
                    fig_q = plot_quantization_axis_vector(
                        quantization_axis,
                        coords=None,
                        title="Quantization axis",
                        show_site_labels=False,
                        elev=q_view_elev,
                        azim=q_view_azim,
                    )
                    safe_tight_layout(fig_q)
                    st.pyplot(fig_q)
                    plt.close(fig_q)
                st.caption(
                    "Drag to rotate the 3D view. The green q-hat arrow is the "
                    "dipolar quantization axis used for every bond-angle factor."
                )

            st.markdown(
                "**Angle convention.** `θ` rotates the planar ladder around the "
                "z-axis. `α` and `β` are the in-plane zigzag angles, and "
                "`θ_q`, `φ_q` define the 3D quantization axis."
            )
            st.divider()
            render_kinetic_diagnostics(
                coords, Jij_full, quantization_axis, L,
                N, hilbert_dim, J01, J02, J13, J03, J12,
            )

        with pi_map_tab:
            st.subheader("Mapping from tilted dipolar chain to π-phase ladder")
            fig_map, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=FIG_DPI)
            plot_normalized_couplings(
                coords,
                Jij_full,
                L,
                ax=axes[0],
                title="Current dipolar chain projection",
                label_digits=2,
                label_scale=0.18,
                show_legend=False,
                show_bond_labels=False,
                site_label_mode="spin",
                use_coupling_colormap=True,
                show_colorbar=True,
            )
            plot_pi_phase_lattice_mapping(
                Jij_full,
                L,
                ax=axes[1],
                title="Effective nearest-neighbor π-phase ladder",
                label_digits=2,
                show_coupling_values=True,
                show_phase_labels=True,
            )
            safe_tight_layout(fig_map)
            st.pyplot(fig_map)
            plt.close(fig_map)
            st.caption(
                "The right panel keeps only the abstract ladder nearest-neighbor "
                "bonds. A negative leg coupling is represented as a π phase "
                "relative to a positive leg coupling."
            )

    # -------------------------------------------------------------------------
    # Clean dynamics tab.
    # -------------------------------------------------------------------------
    with dynamics_tab:
        st.subheader("Spin dynamics: exact diagonalization via QuSpin")
        controls = st.columns([1.2, 1.2, 1.6])
        with controls[0]:
            n_pi = st.slider(
                "Total time (× π / |J_02|)", 1, 12, 8,
                key="dyn_n_pi",
            )
        with controls[1]:
            num_points = st.slider(
                "Number of time points", 100, 2000, 800, step=100,
                key="dyn_num_points",
            )
        with controls[2]:
            interaction_choice = st.radio(
                "Interaction model(s)",
                [
                    "compare both",
                    "only nearest neighbor interaction",
                    "all-pair dipolar interaction",
                ],
                horizontal=False,
                key="dyn_model_choice",
            )
        show_nearest, show_all_pair = model_flags(interaction_choice)
        run = st.button("Run dynamics", type="primary", key="run_dynamics")

        if run:
            with st.spinner("Running QuSpin time evolution."):
                st.session_state["dyn_result"] = simulate_ladder(
                    L=L,
                    alpha_deg=alpha_deg,
                    beta_deg=beta_deg,
                    theta_deg=theta_geo_deg,
                    quantization_axis=quantization_axis,
                    n_pi=float(n_pi),
                    num_points=int(num_points),
                    include_full=show_all_pair,
                    include_designed=show_nearest,
                )
                st.session_state["dyn_show_nearest"] = show_nearest
                st.session_state["dyn_show_all_pair"] = show_all_pair

        result = st.session_state.get("dyn_result")
        if result is not None:
            x_axis = np.array(result["x_axis"], dtype=float)
            if (
                "dyn_cut_x" not in st.session_state
                or st.session_state["dyn_cut_x"] < 0.0
                or st.session_state["dyn_cut_x"] > float(x_axis[-1])
            ):
                st.session_state["dyn_cut_x"] = float(0.5 * x_axis[-1])

            cut_x = st.slider(
                "Line cut time |J_02| t",
                0.0,
                float(x_axis[-1]),
                key="dyn_cut_x",
                step=max(float(x_axis[-1]) / 500.0, 1e-3),
            )
            cut_index = int(np.argmin(np.abs(x_axis - cut_x)))
            cut_x_actual = float(x_axis[cut_index])

            fig_dyn = make_imbalance_fidelity_figure(
                result,
                show_designed=st.session_state.get("dyn_show_nearest", show_nearest),
                show_full=st.session_state.get("dyn_show_all_pair", show_all_pair),
                cut_x=cut_x_actual,
            )
            safe_tight_layout(fig_dyn)
            st.pyplot(fig_dyn)
            plt.close(fig_dyn)

            spin_options = []
            if "site_mag_full" in result:
                spin_options.append("all-pair dipolar interaction")
            if "site_mag_designed" in result:
                spin_options.append("only nearest neighbor interaction")
            if st.session_state.get("dyn_spin_source") not in spin_options:
                st.session_state["dyn_spin_source"] = spin_options[0]
            spin_source = st.radio(
                "Spin configuration line cut",
                spin_options,
                horizontal=True,
                key="dyn_spin_source",
            )
            site_mag_key = (
                "site_mag_full"
                if spin_source == "all-pair dipolar interaction"
                else "site_mag_designed"
            )
            fig_spin = make_spin_configuration_figure(
                result["coords"],
                result[site_mag_key][cut_index],
                title=rf"Spin configuration at $|J_{{02}}|t = {cut_x_actual:.2f}$",
            )
            safe_tight_layout(fig_spin)
            st.pyplot(fig_spin)
            plt.close(fig_spin)
            st.caption(
                f"Initial state {result['state_str']} "
                f"(top sites |down>, bottom sites |up>) · "
                f"|J_02| = {abs(result['J02']):.4f} · "
                f"line cut uses point {cut_index + 1} of {len(x_axis)}."
            )
        else:
            st.info(
                "Choose the interaction model and click Run dynamics. The "
                "display/mapping tab updates live without re-running ED."
            )

    # -------------------------------------------------------------------------
    # z-position disorder tab. All noise controls live here, not in sidebar.
    # -------------------------------------------------------------------------
    with disorder_tab:
        st.subheader("z-position disorder ensemble")
        st.markdown(
            "This samples iid Gaussian displacement along the z/proxy tweezer "
            "propagation axis and recomputes the all-pair dipolar couplings "
            "using the clean geometry as the fixed energy scale."
        )
        noise_controls = st.columns([1.1, 1.1, 1.1, 1.2])
        with noise_controls[0]:
            sigma_z = st.slider(
                "σ_z (units of dx)", 0.0, 0.20, 0.02, step=0.005,
                key="noise_sigma_z",
            )
        with noise_controls[1]:
            n_shots = st.slider(
                "Disorder shots", 1, 30, 8,
                key="noise_n_shots",
            )
        with noise_controls[2]:
            seed = st.number_input(
                "RNG seed", min_value=0, max_value=10_000,
                value=1, step=1, key="noise_seed",
            )
        with noise_controls[3]:
            noise_num_points = st.slider(
                "Time points", 100, 1200, 600, step=100,
                key="noise_num_points",
            )
        noise_n_pi = st.slider(
            "Total time for disorder run (× π / |J_02|)",
            1, 12, 8, key="noise_n_pi",
        )
        run_noise = st.button("Run disorder ensemble", key="run_noise")

        if run_noise:
            if L > 5:
                st.warning(
                    f"L = {L} with {n_shots} shots may be slow; consider L ≤ 5."
                )
            with st.spinner(
                f"Running {n_shots} z-disorder shots with σ_z = {sigma_z:.3f}."
            ):
                noise_result = simulate_noise_ensemble(
                    L=L,
                    sigma_z=float(sigma_z),
                    n_shots=int(n_shots),
                    seed=int(seed),
                    alpha_deg=alpha_deg,
                    beta_deg=beta_deg,
                    theta_deg=theta_geo_deg,
                    quantization_axis=quantization_axis,
                    n_pi=float(noise_n_pi),
                    num_points=int(noise_num_points),
                )
            fig_noise = make_noise_figure(noise_result)
            safe_tight_layout(fig_noise)
            st.pyplot(fig_noise)
            plt.close(fig_noise)
            st.caption(
                "Gray traces: individual disorder realizations. Solid: ensemble "
                "average ± 1σ. Dashed: clean all-pair dipolar reference."
            )
        else:
            st.info("Set disorder parameters here, then click Run disorder ensemble.")


# =============================================================================
# About page
# =============================================================================

else:
    st.title("About this app")
    st.markdown(
        """
        This is a compact Streamlit UI for the coordinate-based dipolar ladder
        QuSpin simulation.

        **Main model.** The helper module constructs all-pair dipolar couplings
        from the current 3D coordinates and quantization axis, normalizes by
        `|J_02|`, and optionally truncates to the abstract nearest-neighbor
        ladder graph.

        **Tabs.** The ladder page separates the static display/mapping, clean
        imbalance/fidelity dynamics, and z-position disorder ensemble into
        separate tabs so the sidebar is reserved for geometry and q-axis
        parameters.

        **File map**

        * `app_qspin.py` -- Streamlit UI.
        * `qspin_dipoles.py` -- coupling construction, QuSpin evolution, and
          plotting helpers.
        * `geometry.py` -- coordinate construction and raw dipole interaction
          utilities expected by `qspin_dipoles.py`.
        """
    )
