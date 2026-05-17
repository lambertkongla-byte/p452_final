from setuptools import setup, find_packages

setup(
    name="p452_final",
    version="0.1.0",
    author="Michael Vayninger",
    author_email="lambertkongla@gmail.com",
    description="Dipolar spin chain simulation with Trotterized quantum circuits",
    packages=find_packages(),
    py_modules=["geometry"],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "qiskit",
    ],
)