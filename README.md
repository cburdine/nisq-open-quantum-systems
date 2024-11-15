# nisq-open-quantum-systems
A Jupyter Notebook and Python codebase repository for the simulation of open quantum systems.

---

## Installing Dependencies

To run these Jupyter notebooks, you will need to first install the dependencies in the `requirements.txt` file.
It is recommended that you do this in a Python virtual environment or a clean pip environment. To install the
dependencies, run the command
```
pip install -r requirements.txt
```
in this directory.

## Running Jupyter Notebooks

You can open the Jupyter notebooks in this directory using JupyterLab or a similar tool. I prefer to use the
regular Jupyter notebook server which you can start by running
```
jupyter-notebook .
```
in this directory.

## Getting Started

To get started, it is recommeneded that you take a look at the "Damped QHO Demo.ipynb" notebook.

## Running Quantum Computation Experiments

In this directory there are three notebooks ending with "Experiments". Executing these notebooks on your preferred
quantum computation provider will require some configuration to submit the compute jobs
to a real quantum backend. (You will need to configure the cell below where it says "Configure
Job Submission Here". Currently, these functions are configured
to use the qiskit-aer simulator backend as an example.)

It is recommended that you start by running the experiments in the "Pauli Channel Experiments.ipynb" notebook,
which contains some basic experiments with relatively few circuits that need to executed. Once those complete
sucessfully, you can do the Schwinger Oscillator and QHO Experiments notebooks, which contain multiple experiments.
