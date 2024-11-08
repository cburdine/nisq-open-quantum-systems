# nisq-open-quantum-systems
A Jupyter Notebook and Python codebase repository for the paper "Efficient Simulation of Open Quantum Systems on NISQ Devices" 

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

## Running Quantum Computation Experiments

In this directory there are three notebooks ending with "Experiments". Executing these notebooks on your preferred
quantum computation provider will require some configuration to submit the compute jobs
to a real quantum backend. Specifically, you will need to configure the cell below where it says "Configure
Job Submission Here". In the notebook there are instructions for how do do this. Essentially, you will need to
configure some variables so that the Qiskit compiler knows which backend to target. Two functions are also exposed
for submitting and retrieving the results of executing Qiskit circuits on real hardware. If your available
hardware backend supports Qiskit primitives, you may also need to install those and import the necessary backend
provider primitives and invoke them in these two exposed functions. Currently, these functions are configured
to use the qiskit-aer simulator backend as an example.

It is recommended that you start by running the experiments in the "Pauli Channel Experiments.ipynb" notebook,
which contains some basic experiments with relatively few circuits that need to executed. Once those complete
sucessfully, you can do the Schwinger Oscillator and QHO Experiments notebooks, which contain multiple experiments.
