from geometry import (
    coords_for_zigzag_chain,
    compute_dipole_interaction,
    indices_nearest_neighbors,
)
import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit.quantum_info import Statevector, SparsePauliOp

magic_angle = 48.2
n_pairs = 3
coords = coords_for_zigzag_chain(n_pairs=n_pairs, theta_deg=magic_angle)

spin_up = [i * 2 + 1 for i in range(n_pairs)]
spin_down = [i * 2 for i in range(n_pairs)]


def create_trotter_circuit(
    coords,
    neighbors,
    spin_up,
    spin_down,
    quantization_axis,
    n_trotter_steps,
    total_time=1.0,
):
    """Create a trotterized circuit for the Heisenberg model for a dipolar zigzag chain.

    Implements the first-order Suzuki-Trotter decomposition of the Heisenberg
    Hamiltonian H = Σ_{i<j} J_ij (S_x^i S_x^j + S_y^i S_y^j + S_z^i S_z^j),
    where J_ij = compute_dipole_interaction(coords[i], coords[j], quantization_axis)
    and S_α = σ_α / 2.

    Each Trotter step applies e^{-i J_ij dt/4 * XX} e^{-i J_ij dt/4 * YY} e^{-i J_ij dt/4 * ZZ}
    per pair, using Qiskit's RXX/RYY/RZZ gates (RXX(θ) = e^{-i θ/2 XX}).

    Args:
        coords (list[tuple[int, int, int]]): A list of 3D coordinates for each spin in the chain
        neighbors (list[list[int]]): A list of lists, where each inner list contains the indices of neighboring spins for the corresponding spin
        spin_up (list[int]): A list of indices for spins initialized in the up state
        spin_down (list[int]): A list of indices for spins initialized in the down state
        quantization_axis (tuple[int, int, int]): The axis along which to quantify the spins
        n_trotter_steps (int): The number of Trotter steps to include in the circuit
        total_time (float): The total evolution time (default 1.0)

    Returns:
        qiskit.QuantumCircuit: The trotterized circuit for the Heisenberg model
    """
    n_qubits = len(coords)
    circuit = qiskit.QuantumCircuit(n_qubits)

    # |0⟩ = spin-up, |1⟩ = spin-down
    for i in spin_down:
        circuit.x(i)

    dt = total_time / n_trotter_steps

    # Precompute dipolar coupling strengths J_ij for all neighbor pairs
    couplings = {}
    for i, neighbors_i in enumerate(neighbors):
        for j in neighbors_i:
            if j > i:
                couplings[(i, j)] = compute_dipole_interaction(
                    coords[i], coords[j], quantization_axis=quantization_axis
                )

    # First-order Trotter: repeat n_trotter_steps times
    # e^{-i H dt} ≈ Π_{i<j} e^{-i J_ij/4 XX dt} e^{-i J_ij/4 YY dt} e^{-i J_ij/4 ZZ dt}
    for _ in range(n_trotter_steps):
        for (i, j), J_ij in couplings.items():
            # RXX/RYY/RZZ(θ) = e^{-i θ/2 P⊗P}, so θ = J_ij * dt / 2
            angle = J_ij * dt / 2
            circuit.rxx(angle, i, j)
            circuit.ryy(angle, i, j)
            circuit.rzz(angle, i, j)

    return circuit


def plot_spin_imbalance(
    coords,
    neighbors,
    spin_up,
    spin_down,
    quantization_axis,
    n_trotter_steps=10,
    times=None,
):
    """Simulate the Heisenberg model and plot site magnetizations and staggered imbalance over time.

    The staggered (sublattice) imbalance I(t) = (1/N) Σ_i ε_i ⟨Z_i⟩, where
    ε_i = +1 for initially spin-up sites and -1 for initially spin-down sites,
    starts at 1 and decays as the spins equilibrate. Total magnetization is
    conserved by the Heisenberg Hamiltonian and is not shown.

    Args:
        coords: 3D coordinates for each spin
        neighbors: adjacency list — neighbors[i] is a list of neighbor indices for spin i
        spin_up: indices of spins initialized in the up state
        spin_down: indices of spins initialized in the down state
        quantization_axis: axis for computing dipole interaction strengths
        n_trotter_steps: Trotter steps per time point (controls accuracy)
        times: array of time points to evaluate (default: 50 points from 0 to 5)
    """
    if times is None:
        times = np.linspace(0, 5, 50)

    n_qubits = len(coords)

    # ε_i encodes the initial spin pattern: +1 for up, -1 for down
    sublattice = np.ones(n_qubits)
    for i in spin_down:
        sublattice[i] = -1

    # Build Pauli-Z operators for each qubit (Qiskit little-endian ordering)
    z_ops = [
        SparsePauliOp("I" * (n_qubits - i - 1) + "Z" + "I" * i) for i in range(n_qubits)
    ]

    site_mag = np.zeros((len(times), n_qubits))
    for t_idx, t in enumerate(times):
        circuit = create_trotter_circuit(
            coords,
            neighbors,
            spin_up,
            spin_down,
            quantization_axis,
            n_trotter_steps=n_trotter_steps,
            total_time=t,
        )
        sv = Statevector(circuit)
        for i in range(n_qubits):
            site_mag[t_idx, i] = sv.expectation_value(z_ops[i]).real

    staggered_imbalance = site_mag @ sublattice / n_qubits

    _, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for i in range(n_qubits):
        label = f'site {i} ({"↑" if sublattice[i] > 0 else "↓"})'
        axes[0].plot(times, site_mag[:, i], label=label)
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axes[0].set_ylabel(r"$\langle Z_i \rangle$")
    axes[0].set_title("Site magnetization")
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].plot(times, staggered_imbalance, color="tab:purple")
    axes[1].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$\frac{1}{N}\sum_i \varepsilon_i \langle Z_i \rangle$")
    axes[1].set_title("Staggered spin imbalance")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    magic_angle = 48.2
    n_pairs = 3
    quantization_axis = (0, np.cos(np.radians(30)), np.sin(np.radians(30)))

    coords = coords_for_zigzag_chain(n_pairs=n_pairs, theta_deg=magic_angle)
    spin_up = [i * 2 + 1 for i in range(n_pairs)]
    spin_down = [i * 2 for i in range(n_pairs)]

    # Convert neighbor pairs to adjacency list expected by create_trotter_circuit
    pairs = indices_nearest_neighbors(coords)
    n_qubits = len(coords)
    neighbors = [[] for _ in range(n_qubits)]
    for i, j in pairs:
        neighbors[i].append(j)
        neighbors[j].append(i)

    times = np.linspace(0, 5, 60)
    plot_spin_imbalance(
        coords,
        neighbors,
        spin_up,
        spin_down,
        quantization_axis,
        n_trotter_steps=10,
        times=times,
    )
