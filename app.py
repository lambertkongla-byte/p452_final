"""
Top-level entry point for the combined Streamlit demo.

Run from the project root:
    streamlit run app.py

Two pages share one sidebar selector:
  • Trotter Simulation         → circuits/app.py
        (sub-tabs: spin dynamics, single-step circuit diagram)
  • Dipolar Ladder (QuSpin)    → qspin/app_qspin.py
        (sub-tabs: geometry display, clean dynamics, position disorder)

Each sub-app keeps its own logic file (circuits/trotter_circuit.py,
qspin/qspin_dipoles.py) and can also be run standalone:
    streamlit run circuits/app.py
    streamlit run qspin/app_qspin.py
"""
import sys
from pathlib import Path

import streamlit as st

# Absolute paths — st.Page can be picky about relative paths (especially when
# the project lives in a directory whose name contains spaces).
HERE = Path(__file__).resolve().parent

# Make geometry.py and both sub-packages importable from the sub-app files.
for sub in ("", "circuits", "qspin"):
    p = str(HERE / sub) if sub else str(HERE)
    if p not in sys.path:
        sys.path.insert(0, p)

# set_page_config must be called exactly once, here, before any other st.* call.
st.set_page_config(
    page_title="Quantum Many-Body Scar Demos",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar branding above the page picker.
with st.sidebar:
    st.title("QMBS Demos")
    st.caption(
        "Companion code for kinetic-frustration scars "
        "(Ding, Verresen & Yan, arXiv:2603.11191)."
    )
    st.divider()

# Build absolute paths to the two sub-apps, and fail loudly with a helpful
# message if either file is missing (Streamlit's own error message is terse).
trotter_path = HERE / "circuits" / "app.py"
qspin_path   = HERE / "qspin"    / "app_qspin.py"

for path in (trotter_path, qspin_path):
    if not path.is_file():
        st.error(
            f"Expected file not found: `{path}`.\n\n"
            f"Project root resolved to `{HERE}`. "
            "Make sure `app.py` sits alongside the `circuits/` and `qspin/` "
            "folders, and that you launched Streamlit from the project root."
        )
        st.stop()

pages = [
    st.Page(
        trotter_path,
        title="Trotter Simulation",
        icon="🌀",
        default=True,
    ),
    st.Page(
        qspin_path,
        title="Dipolar Ladder (QuSpin)",
        icon="🧲",
    ),
]

pg = st.navigation(pages)
pg.run()
