import unittest

from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

import numpy as np

from open_quantum_systems.tomography import KrausSeriesSampler, KrausSeriesEstimator, KrausSeriesTomography


class TestKrausSeriesSampler(unittest.TestCase):

    def setUp(self):

        psi_a = np.array([ 0, 1.j, 1, 0]) / np.sqrt(2)
        psi_b = np.array([ 1.j, 0, 0, 1]) / np.sqrt(2)

        qc_a = QuantumCircuit(2)
        qc_b = QuantumCircuit(2)
        qc_a.append(StatePreparation(psi_0), [0,1])
        qc_b.append(StatePreparation(psi_b), [0,1])

        self.kraus_circuits = [ qc_a, qc_b ]

        self.backend = Aer.get_backend('qasm_simulator')

    def test_launch_jobs(self):

        sampler = KrausSeriesSampler(self.backend)
        jobs = self.sampler.launch_jobs(self.kraus_circuits)
        
        assert len(jobs) == len(kraus_circuits)
        
        

class TestKrausSeriesEstimtor(unittest.TestCase):
    

class TestKrausSeriesTomography(unittest.TestCase):
    pass

