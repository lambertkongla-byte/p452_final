# p452 Final Project — Dipolar Spin Chain Simulation

Simulates quantum spin dynamics of a dipolar zigzag chain using a first-order Suzuki-Trotter decomposition of the Heisenberg Hamiltonian, implemented with Qiskit quantum circuits.

## Overview

- **Geometry** (`geometry.py`): Generates zigzag chain coordinates at the magic angle, computes dipole-dipole interaction strengths, and finds nearest-neighbor pairs.
- **Trotter Circuit** (`circuits/trotter_circuit.py`): Builds a trotterized Qiskit circuit for time evolution under the dipolar Heisenberg Hamiltonian and plots site magnetizations and staggered spin imbalance over time.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/mvlabtop/p452_final.git
cd p452_final

# (Optional) Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

## Usage

Run the geometry visualization:

```bash
python geometry.py
```

Run the Trotter circuit simulation and plot spin imbalance:

```bash
python circuits/trotter_circuit.py
```

## Physics Background

The Hamiltonian is:

$$H = \sum_{i<j} J_{ij} \left( S_x^i S_x^j + S_y^i S_y^j + S_z^i S_z^j \right)$$

where $J_{ij} \propto (3\cos^2\theta_{ij} - 1)/r_{ij}^3$ is the dipole-dipole coupling. The zigzag geometry at the magic angle (~48.2°) tunes the effective interactions between sublattices.

## Project Structure

```
p452_final/
├── geometry.py              # Zigzag chain geometry and dipole interactions
├── circuits/
│   └── trotter_circuit.py   # Trotterized Qiskit simulation
├── requirements.txt
└── setup.py
```
