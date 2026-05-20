# P452 Final Project — Dipolar Spin Chain Quantum Many-Body Scars

Interactive Streamlit demos and backend simulation code for kinetic-frustration quantum many-body scars (QMBS) in a tilted dipolar zigzag chain, based on [Ding, Verresen & Yan, arXiv:2603.11191](https://arxiv.org/abs/2603.11191).

Two complementary simulation approaches are implemented side-by-side:

| Page | Method | Backend |
|---|---|---|
| **Trotter Simulation** | First-order Suzuki-Trotter on a quantum circuit | Qiskit statevector |
| **Dipolar Ladder (QuSpin)** | Exact diagonalization via the XY spin-exchange Hamiltonian | QuSpin |

---

## Physics Background

### The model

The system is a 2×L "twisted ladder" of spin-1/2 particles arranged in a zigzag chain. Each pair of sites (one on the top leg, one on the bottom leg) forms a rung. The sites interact via the dipole-dipole coupling:

$$J_{ij} = \frac{1 - 3\cos^2\theta_{ij}}{r_{ij}^3}$$

where $\theta_{ij}$ is the angle between the bond $\hat{r}_{ij}$ and the quantization axis $\hat{q}$, and $r_{ij}$ is the inter-site distance.

**Trotter page** evolves the full Heisenberg Hamiltonian:

$$H = \sum_{i<j} J_{ij}\left(S_x^i S_x^j + S_y^i S_y^j + S_z^i S_z^j\right)$$

**QuSpin page** evolves the spin-exchange (XY) Hamiltonian:

$$H_\mathrm{SE} = \sum_{i<j} \frac{J_{ij}}{2}\left(S_i^+ S_j^- + S_i^- S_j^+\right)$$

### Kinetic-frustration scar

The zigzag geometry is tuned so that the two plaquette-diagonal couplings satisfy $J_{02} \approx -J_{13}$ (equal magnitude, opposite sign). At the **dipolar magic angle** $\theta_\mathrm{magic} \approx 54.74°$ ($\cos^{-1}(1/\sqrt{3})$), the plaquette-crossing couplings $J_{03}$ and $J_{12}$ are suppressed. Under these conditions the Néel-like product state $|{\downarrow\uparrow\downarrow\uparrow\cdots}\rangle$ becomes a kinetic-frustration scar: it oscillates coherently at a frequency set by $|J_{02}|$ without thermalizing, even though it is an excited state of a non-integrable Hamiltonian.

The **staggered spin imbalance**

$$\mathcal{I}(t) = \frac{1}{N}\sum_i \varepsilon_i \langle Z_i(t)\rangle, \qquad \varepsilon_i = \langle Z_i(0)\rangle$$

stays close to 1 for scar states and decays to 0 for thermal states.

### In-plane geometry

The zigzag chain is built from the angles $\alpha$ and $\beta$ (the two in-plane bond angles at each vertex) and an overall in-plane rotation $\theta$. The quantization axis $\hat{q}$ is parameterized by its spherical angles $(\theta_q, \varphi_q)$. The "magic angle" button in the QuSpin app auto-searches for the $\theta$ that brings the plaquette-diagonal bonds closest to the dipolar magic angle for the current $\hat{q}$.

---

## Project Structure

```
p452_final/
├── app.py                      # Top-level Streamlit entry point (multi-page)
├── geometry.py                 # Zigzag chain coordinates and dipole couplings
├── requirements.txt
├── setup.py
│
├── circuits/
│   ├── app.py                  # Streamlit page: Trotter circuit UI
│   └── trotter_circuit.py      # Qiskit Trotter circuit builder and spin dynamics
│
└── qspin/
    ├── app_qspin.py            # Streamlit page: QuSpin exact-diagonalization UI
    └── qspin_dipoles.py        # QuSpin Hamiltonian, evolution, and plotting helpers
```

### Module summary

**`geometry.py`**
- `coords_for_zigzag_chain` — generates 3D coordinates for a 2×L zigzag chain given angles $\alpha$, $\beta$, and in-plane rotation $\theta$.
- `compute_dipole_interaction` — returns $J_{ij} = (3\cos^2\theta_{ij}-1)/r_{ij}^3$ for any bond.
- `indices_nearest_neighbors` / `indices_next_nearest_neighbors` — distance-threshold neighbor finders.

**`circuits/trotter_circuit.py`**
- `create_trotter_circuit` — builds a Qiskit circuit implementing the first-order Suzuki-Trotter decomposition; uses `RXX`, `RYY`, `RZZ` gates.
- `compute_spin_imbalance` — sweeps over a time array, runs Statevector simulation at each point, and returns site magnetizations and the staggered imbalance.
- `make_figure` — two-panel matplotlib figure (site magnetization + staggered imbalance).

**`circuits/app.py`** (Trotter Streamlit page)
- Sidebar controls for chain size, magic angle, quantization axis, z-position disorder, NNN interactions, Trotter steps, and time range.
- Live chain geometry plot.
- Sim A / Sim B comparison panel with overlay and difference plots.
- Single-step circuit diagram rendered via Qiskit's matplotlib drawer.

**`qspin/qspin_dipoles.py`**
- `build_all_dipolar_couplings_from_geometry` — all-pair $J_{ij}$ normalized to $|J_{02}|=1$.
- `keep_designed_bonds_only` — trims to the abstract nearest-neighbor (rung + leg) ladder graph.
- `build_quspin_xy_hamiltonian` — wraps couplings into a QuSpin `hamiltonian` object.
- `simulate_ladder` — high-level driver: builds basis, evolves from $|01\,01\cdots\rangle$, computes imbalance and fidelity for both full dipolar and nearest-neighbor models.
- `simulate_noise_ensemble` — runs multiple shots with iid Gaussian z-position disorder.
- Plotting: `plot_normalized_couplings`, `plot_pi_phase_lattice_mapping`, `plot_quantization_axis_vector`, `make_imbalance_fidelity_figure`, `make_noise_figure`, `make_spin_configuration_figure`.

**`qspin/app_qspin.py`** (QuSpin Streamlit page)
- Three tabs: **Display / mapping** (geometry + $\hat{q}$ 3D view + $\pi$-phase ladder mapping), **Imbalance and fidelity** (exact dynamics, line-cut spin snapshot), **z-position disorder ensemble** (noise averaging).
- Sidebar: chain size $L$, angles $\alpha$/$\beta$/$\theta$, quantization axis $(\theta_q,\varphi_q)$, magic-angle auto-set button.
- Kinetic-frustration diagnostics: magic-angle check, $J_{02}+J_{13}$ cancellation, diagonal leakage, Hilbert-space dimension.

---

## Installation

### Prerequisites

- Python 3.10+
- conda (recommended) or pip

### Steps

```bash
# Clone the repository
git clone https://github.com/mvlabtop/p452_final.git
cd p452_final

# (Recommended) create a conda environment
conda create -n p452 python=3.11
conda activate p452

# QuSpin is most reliably installed via conda
conda install -c weinbe58 quspin

# Install remaining dependencies
pip install -r requirements.txt

# Install the package in editable mode (makes geometry importable everywhere)
pip install -e .
```

> **Note on QuSpin:** `pip install quspin` builds from source and may fail on some systems. If it does, use the conda channel above and comment out the `quspin` line in `requirements.txt`.

---

## Usage

### Combined Streamlit app (recommended)

```bash
streamlit run app.py
```

Opens both pages in one browser window with a shared sidebar selector.

### Individual pages

```bash
# Trotter circuit page only
streamlit run circuits/app.py

# QuSpin exact-diagonalization page only
streamlit run qspin/app_qspin.py
```

### Standalone scripts

```bash
# Geometry visualization with dipole coupling labels
python geometry.py

# Trotter simulation: site magnetization + staggered imbalance plot
python circuits/trotter_circuit.py

# QuSpin exact dynamics: imbalance and fidelity for L=3
python qspin/qspin_dipoles.py
```

---

## Key Parameters

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| Spin pairs | $L$ | 3 | Number of rungs; total sites $N = 2L$ |
| In-plane zigzag angles | $\alpha$, $\beta$ | 34°, 32.45° | Bond angles at each vertex |
| In-plane rotation | $\theta$ | 48.2° | Rotation of ladder around z-axis |
| Quantization axis polar angle | $\theta_q$ | 60° | Angle of $\hat{q}$ from z-axis |
| Quantization axis azimuth | $\varphi_q$ | 90° | Azimuth of $\hat{q}$ in xy-plane |
| Trotter steps | — | 10 | First-order Trotter steps per time point |
| z-position noise | $\sigma_z$ | 0 | Gaussian disorder std. dev. (units of $dx$) |

The **magic-angle condition** requires the plaquette-diagonal bonds to satisfy $|\hat{q}\cdot\hat{r}_{ij}| = 1/\sqrt{3}$, i.e. $\theta_{ij} = \cos^{-1}(1/\sqrt{3}) \approx 54.74°$. The "Set θ to magic angle" button in the QuSpin app searches for the $\theta$ that satisfies this condition numerically for the current $\hat{q}$, $\alpha$, and $\beta$.

---

## Reference

> Ding, Verresen & Yan, *Exact quantum scars from kinetic frustration for cross-platform realizations*, [arXiv:2603.11191](https://arxiv.org/abs/2603.11191) (2026).
