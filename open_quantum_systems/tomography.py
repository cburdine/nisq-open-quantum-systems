from collections.abc import Iterable

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

import numpy as np

from .math_util import *

def to_int_result_dict(results):

    int_result_dict = {}
    for k, n in results.items():
        i = int(k,2) if isinstance(k, str) else int(k)
        int_result_dict[i] = n

    return int_result_dict

def to_density_result_dict(results, n_qubits):
    
    density_result_dict = {}
    for k, n in results.items():
        if isinstance(k, str):
            i, j = int(k[:n_qubits],2), int(k[n_qubits:],2)
        else:
            i, j = k>>n_qubits, k&((1<<n_qubits)-1)
        density_result_dict[(i,j)] = n
    
    return density_result_dict


def get_conditional_result(results, conditional_qubits):
    cond_bits = 0
    for qubit in conditional_qubits:
        cond_bits |= (1<<qubit)

    return {
        k : n for k,n in results.items() if not (k & cond_bits)
    }
        

def density_trace(results):
    return sum([p for (i,j), p in results.items() if i == j])


def apply_result_mask(counts, result_mask):
    return  { k : n for k,n in counts.items() if result_mask[k] }

def apply_density_result_mask(counts, result_mask):
    result_mask = np.array(result_mask)
    if result_mask.ndim >= 1:
        return { (i,j) : n for (i,j),n in counts.items() if result_mask[i,j] }
    else:
        return { (i,j) : n for (i,j),n in counts.items() if result_mask[i] and result_mask[j] }
    

def basis_bitmask(basis_str):
    bitmask = 0
    for ch in basis_str:
        bitmask <<= 1
        if ch != 'I':
            bitmask |= 1
    
    return bitmask

def subbasis_str_from_bitmask(bitmask, basis_str):
    n = len(basis_str)
    subbasis_str = ['I']*n

    for i, ch in enumerate(basis_str):
        if (bitmask>>(n-(i+1)))&1:
            subbasis_str[i] = ch

    return ''.join(subbasis_str)


def basis_measurement_parity(n, bitmasks):

    n = (n&bitmasks)
    parities = np.zeros_like(bitmasks)
    while np.any(n):
        parities ^= (n&1)
        n>>=1
    return parities

def make_basis_measurement_circuit(basis_str, conj=False):
    qc = QuantumCircuit(len(basis_str))
    for i, ch in enumerate(reversed(basis_str)):
        if ch == 'X':
            qc.h(i)
        elif ch == 'Y':
            if conj:
                qc.u(np.pi/2, 0, -np.pi/2, i)
            else:
                qc.u(np.pi/2, 0,  np.pi/2, i)
    
    return qc

class KrausSeriesPrimitive:

    def __init__(self, qiskit_backend,
                 qiskit_transpile_options={},
                 qiskit_run_options={},
                 qiskit_result_options={},
                 transpile_fn=None,
                 launch_job_fn=None,
                 get_result_fn=None):

        self.qiskit_backend = qiskit_backend
        self.qiskit_transpile_options = qiskit_transpile_options
        self.qiskit_run_options = qiskit_run_options
        self.qiskit_result_options = qiskit_result_options
        self.transpile_fn = transpile_fn
        self.launch_job_fn = launch_job_fn
        self.get_result_fn = get_result_fn

    def _format_parameters(self, parameters=None, n_circuits=1):

        if parameters is None:
            return np.array([[ None ]*n_circuits ])
        
        params_np = np.array(parameters)

        if params_np.ndim > 3:
            raise ValueError(f"Expected Parameters of dimension at most 3 (got shape: {params_np.shape})")
        elif params_np.ndim == 3:
            if params_np.shape[1] != n_circuits:
                raise ValueError(f"Expected Parameters with shape matching (*,{n_circuits},*). (got shape: {params_np.shape})")
        elif params_np.ndim == 2:
            if params_np.dtype == np.dtype('O'):
                if params_np.shape[1] != n_circuits:
                    raise ValueError(f"Expected Parameters with shape matching (*,{n_circuits}). (got shape: {params_np.shape})")
                params_np = params_np.reshape((-1, n_circuits))
            else:
                params_np = params_np.reshape((1, n_circuits, -1))
        elif params_np.ndim == 1:
            if params_np.shape[0] != n_circuits:
                raise ValueError(f"Expected Parameters with shape matching ({n_circuits},). (got shape: {params_np.shape})")
            params_np = params_np.reshape((1,1, -1))
        else:
            if n_circuits != 1:
                raise ValueError(f"Expected Parameters with shape at least matching ({n_circuits},). (got shape: {params_np.shape})")
            params_np = params_np.reshape((1,1,1))
        
        return params_np

    def _bind_parameters(self, circuit, parameters, series_idx):
        
        bound_circuits = []
        for p in parameters:
            if parameters is not None and len(p) > 0 and p[series_idx] is not None:
                bound_circuit = circuit.assign_parameters(p[series_idx])
            else:
                bound_circuit = circuit
            bound_circuits.append(bound_circuit)
        
        return bound_circuits

    def _transpile(self, circuits):
        if self.transpile_fn is not None:
            trans_circuits = self.transpile_fn(circuits)
        else:
            trans_circuits = transpile(circuits, self.qiskit_backend)
        
        return trans_circuits

    def _launch_job(self, circuits, shots):
        if self.launch_job_fn is not None:
            job = self.launch_job_fn(circuits, shots)
        else:
            # Note: the execute() function has been deprecated in favor of runtime primitives in v1.0:
            # job = execute(circuits, backend=self.qiskit_backend, shots=shots, **self.qiskit_run_options)
            job = self.qiskit_backend.run(circuits, shots=shots, **self.qiskit_run_options)
        
        return job

    def _get_job_results(self, job):
        if self.get_result_fn is not None:
            result_counts = self.get_result_fn(job)
            result_int_counts = [ to_int_result_dict(res) for res in result_counts ]
        else:
            result_counts = job.result(**self.qiskit_result_options).get_counts()
            if isinstance(result_counts, list):
                result_int_counts = [ res.int_outcomes() for res in result_counts ]
            else:
                result_int_counts = [ result_counts.int_outcomes() ]

        return result_int_counts

class KrausSeriesSampler(KrausSeriesPrimitive):

    def __init__(self, qiskit_backend, **kwargs):
        super().__init__(qiskit_backend, **kwargs)
        
    def launch_jobs(self,
            circuits,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):
        
        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))

        # submit jobs:
        job_array = []
        for i, circuit in enumerate(circuits):

            # build measurement circuit:
            if conditional_qubits is None:
                conditional_qubits = []
            
            measurement_qubits = observable_qubits+conditional_qubits
            meas_circuit = QuantumCircuit(len(circuit.qubits), len(measurement_qubits),
                                name=circuit.name+'_samp')
            meas_circuit.append(circuit.decompose(), list(range(len(circuit.qubits))))
            meas_circuit.measure(measurement_qubits,list(range(len(measurement_qubits))))

            # bind parameters:
            bound_circuits = self._bind_parameters(meas_circuit, parameters, series_idx=i)

            # transpile and run circuits:
            trans_circuits = self._transpile(bound_circuits)
            job = self._launch_job(trans_circuits, shots_per_circuit)
            job_array.append(job)

        return job_array

    def get_results(self,
            job_array,
            circuits,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):

            if not isinstance(circuits, Iterable):
                circuits = [circuits]
            assert len(circuits) == len(job_array), "circuits and job_array are mismatched."

            # create sample probabilities list:
            parameters = self._format_parameters(parameters, len(circuits))
            job_sample_probabilities = [ {} for _ in range(parameters.shape[0])]

            # retrieve results from job array:
            for i, circuit in enumerate(circuits):

                # get measurement and observable qubits:
                if conditional_qubits is None:
                    conditional_qubits = []
                measurement_qubits = observable_qubits+conditional_qubits

                job = job_array[i]
                job_counts = self._get_job_results(job)

                for j, (sample_probs, counts) in enumerate(zip(job_sample_probabilities, job_counts)):

                    # get conditional results:
                    if conditional_qubits:
                        conditional_qubit_idxs = [ measurement_qubits.index(q) for q in conditional_qubits ]
                        counts = get_conditional_result(counts, conditional_qubit_idxs)

                    # mask measured counts:
                    if result_masks is not None:
                        counts = apply_result_mask(counts, result_masks[i])

                    # add results to total probability distribution:
                    weight = 1.0 if circuit_weights is None else circuit_weights[j][i]
                    
                    for k, n in counts.items():
                        if k not in sample_probs:
                            sample_probs[k] = 0.0
                        sample_probs[k] += weight*n/shots_per_circuit

            # Renormalize sample probabilities:
            if renormalize:
                for sample_probs in job_sample_probabilities:
                    total = sum(sample_probs.values())
                    for k in sample_probs:
                        sample_probs[k] /= total

            return job_sample_probabilities

    def _run(self,
            circuits,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):

        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))

        batch_sample_probabilities = [ {} for _ in range(parameters.shape[0])]

        # launch job array (if not given):
        if job_array is None:
            job_array = []
            for i, circuit in enumerate(circuits):

                # build measurement circuit:
                if conditional_qubits is None:
                    conditional_qubits = []
                
                measurement_qubits = observable_qubits+conditional_qubits
                meas_circuit = QuantumCircuit(len(circuit.qubits), len(measurement_qubits))
                meas_circuit.append(circuit.decompose(), list(range(len(circuit.qubits))))
                meas_circuit.measure(measurement_qubits,list(range(len(measurement_qubits))))

                # bind parameters:
                bound_circuits = self._bind_parameters(meas_circuit, parameters, series_idx=i)

                # transpile and run circuits:
                trans_circuits = self._transpile(bound_circuits)
                job = self._launch_job(trans_circuits, shots_per_circuit)
                job_array.append(job)

        # retrieve results of job array:
        for i, circuit in enumerate(circuits):
            job = job_array[i]
            job_counts = self._get_job_results(job)

            for j, (sample_probs, counts) in enumerate(zip(batch_sample_probabilities, job_counts)):

                # get conditional results:
                if conditional_qubits:
                    conditional_qubit_idxs = [ measurement_qubits.index(q) for q in conditional_qubits ]
                    counts = get_conditional_result(counts, conditional_qubit_idxs)

                # mask measured counts:
                if result_masks is not None:
                    counts = apply_result_mask(counts, result_masks[i])

                # add results to total probability distribution:
                weight = 1.0 if circuit_weights is None else circuit_weights[j][i]
                
                for k, n in counts.items():
                    if k not in sample_probs:
                        sample_probs[k] = 0.0
                    sample_probs[k] += weight*n/shots_per_circuit

        # Renormalize sample probabilities:
        if renormalize:
            for sample_probs in batch_sample_probabilities:
                total = sum(sample_probs.values())
                for k in sample_probs:
                    sample_probs[k] /= total

        return batch_sample_probabilities

    def run(self,
            circuits,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):

        job_array = self.launch_jobs(
            circuits=circuits,
            observable_qubits=observable_qubits,
            conditional_qubits=conditional_qubits,
            shots_per_circuit=shots_per_circuit,
            parameters=parameters,
            result_masks=result_masks,
            circuit_weights=circuit_weights,
            renormalize=renormalize)

        job_sample_probabilities = self.get_results(
            job_array=job_array,
            circuits=circuits,
            observable_qubits=observable_qubits,
            conditional_qubits=conditional_qubits,
            shots_per_circuit=shots_per_circuit,
            parameters=parameters,
            result_masks=result_masks,
            circuit_weights=circuit_weights,
            renormalize=renormalize)

        return job_sample_probabilities

class KrausSeriesEstimator(KrausSeriesPrimitive):

    def __init__(self, qiskit_backend, **kwargs):
        super().__init__(qiskit_backend, **kwargs)

    def launch_jobs(self,
            circuits,
            observable,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):
        
        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))

        # mask observables:
        if result_masks is None:
            masked_observables = [observable]*len(circuits)
        else:
            result_masks = [ np.array(m, dtype=np.float64) for m in result_masks ]
            masked_observables = [
                mask.reshape(1,-1)*observable*mask.reshape(-1,1)
                for mask in result_masks
            ]

        job_array = []
        job_pauli_array = []

        for i, circuit in enumerate(circuits):

            # resolve any conditional qubits:
            if conditional_qubits is None:
                conditional_qubits = []

            # Note: if renormalizing, we still need estimates of the final probabilities
            # of the non-environment subsystem, so we use a trivial Pauli dict:
            pauli_dict = to_pauli_dict(masked_observables[i])
            if not pauli_dict:
                if renormalize:
                    pauli_dict = {'I'*len(observable_qubits) : 0.0}
                else:
                    continue
                
            # decompose observable into commuting groups of weighted Pauli strings:
            measurement_pauli_dicts = get_measurement_basis_pauli_dicts(pauli_dict)

            # Submit a job for each measurement basis:
            measurement_jobs = {}
            for basis_str, meas_dict in measurement_pauli_dicts.items():
                
                assert(len(basis_str) == len(observable_qubits))
                
                # build measurement circuit:
                measurement_qubits = observable_qubits+conditional_qubits
                meas_circuit = QuantumCircuit(len(circuit.qubits), len(measurement_qubits),
                                        name=circuit.name+f'_estm_{basis_str.lower()}')
                meas_circuit.append(circuit.decompose(), list(range(len(circuit.qubits))))
                meas_circuit.append(make_basis_measurement_circuit(basis_str), observable_qubits)
                meas_circuit.measure(measurement_qubits, list(range(len(measurement_qubits))))
                
                # bind circuit parameters and execute:
                bound_circuits = self._bind_parameters(meas_circuit, parameters, series_idx=i)
                trans_circuits = self._transpile(bound_circuits)
                job = self._launch_job(trans_circuits, shots=shots_per_circuit)
                measurement_jobs[basis_str] = job

            # record job and associated Pauli dictionary:
            job_array.append(measurement_jobs)
            job_pauli_array.append(measurement_pauli_dicts)
            

        return job_array, job_pauli_array

    def get_results(self,
            job_array,
            job_pauli_array,
            circuits,
            observable,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):
        
        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))

        # get job results for each circuit:
        job_evals = [ 0 for _ in range(parameters.shape[0]) ]
        job_total_probs = [ 0 for _ in range(parameters.shape[0]) ]

        assert len(job_array) == len(circuits)
        assert len(job_pauli_array) == len(circuits)

        for i, circuit in enumerate(circuits):
            
            measurement_pauli_dicts = job_pauli_array[i]
            measurement_jobs = job_array[i]

            for basis_str, meas_dict in measurement_pauli_dicts.items():
                
                assert(len(basis_str) == len(observable_qubits))
                
                # build measurement circuit:
                measurement_qubits = observable_qubits+conditional_qubits
                job = measurement_jobs[basis_str]
                job_counts = self._get_job_results(job)

                for j, counts in enumerate(job_counts):

                    # get conditional results:
                    if conditional_qubits:
                        conditional_qubit_idxs = [ measurement_qubits.index(q) for q in conditional_qubits ]
                        counts = get_conditional_result(counts, conditional_qubit_idxs)

                    # construct bitmask dictionary for each observable in this basis:
                    pauli_strs = list(meas_dict.keys())
                    bitmasks = np.array([ basis_bitmask(p) for p in pauli_strs ])
                    pauli_weights = np.array([ meas_dict[p] for p in pauli_strs ])
                    pauli_evals = np.zeros(len(pauli_strs))
                    
                    total_meas_circuit_counts = 0
                    for k, n in counts.items():
                        parities = (-1)**basis_measurement_parity(k,bitmasks)
                        pauli_evals += parities*n
                        total_meas_circuit_counts += n

                    circuit_weight = 1.0 if circuit_weights is None else circuit_weights[j][i]
                    job_evals[j] += np.sum(pauli_evals * pauli_weights) * circuit_weight / shots_per_circuit
                    job_total_probs[j] += total_meas_circuit_counts * circuit_weight / (len(measurement_pauli_dicts)*shots_per_circuit)

        job_evals = np.array(job_evals)
        job_total_probs = np.array(job_total_probs)

        # renormalize expectation values:
        if renormalize:
            job_evals /= job_total_probs

        return job_evals.tolist()


    def _run(self, circuits,
            observable,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):

        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))
        batch_evals = [ 0 for _ in range(parameters.shape[0]) ]
        batch_total_probs = [ 0 for _ in range(parameters.shape[0]) ]

        # mask observables:
        if result_masks is None:
            masked_observables = [observable]*len(circuits)
        else:
            result_masks = [ np.array(m, dtype=np.float64) for m in result_masks ]
            masked_observables = [
                mask.reshape(1,-1)*observable*mask.reshape(-1,1)
                for mask in result_masks
            ]

        for i, circuit in enumerate(circuits):

            # resolve any conditional qubits:
            if conditional_qubits is None:
                conditional_qubits = []

            # Note: if renormalizing, we still need estimates of the final probabilities
            # of the non-environment subsystem, so we use a trivial Pauli dict:
            pauli_dict = to_pauli_dict(masked_observables[i])
            if not pauli_dict:
                if renormalize:
                    pauli_dict = {'II' : 0.0}
                else:
                    continue
                
            # decompose observable into commuting groups of weighted Pauli strings:
            measurement_pauli_dicts = get_measurement_basis_pauli_dicts(pauli_dict)
            #circuit_weight = 1.0 if circuit_weights is None else circuit_weights[i]
            
            for basis_str, meas_dict in measurement_pauli_dicts.items():
                
                assert(len(basis_str) == len(observable_qubits))
                
                # build measurement circuit:
                measurement_qubits = observable_qubits+conditional_qubits
                meas_circuit = QuantumCircuit(len(circuit.qubits), len(measurement_qubits))
                meas_circuit.append(circuit.decompose(), list(range(len(circuit.qubits))))
                meas_circuit.append(make_basis_measurement_circuit(basis_str), observable_qubits)
                meas_circuit.measure(measurement_qubits, list(range(len(measurement_qubits))))
                
                # bind circuit parameters and execute:
                bound_circuits = self._bind_parameters(meas_circuit, parameters, series_idx=i)
                trans_circuits = self._transpile(bound_circuits)
                job = self._launch_job(trans_circuits, shots=shots_per_circuit)
                job_counts = self._get_job_results(job)

                for j, counts in enumerate(job_counts):

                    # get conditional results:
                    if conditional_qubits:
                        conditional_qubit_idxs = [ measurement_qubits.index(q) for q in conditional_qubits ]
                        counts = get_conditional_result(counts, conditional_qubit_idxs)

                    # construct bitmask dictionary for each observable in this basis:
                    pauli_strs = list(meas_dict.keys())
                    bitmasks = np.array([ basis_bitmask(p) for p in pauli_strs ])
                    pauli_weights = np.array([ meas_dict[p] for p in pauli_strs ])
                    pauli_evals = np.zeros(len(pauli_strs))
                    
                    total_meas_circuit_counts = 0
                    for k, n in counts.items():
                        parities = (-1)**basis_measurement_parity(k,bitmasks)
                        pauli_evals += parities*n
                        total_meas_circuit_counts += n

                    circuit_weight = 1.0 if circuit_weights is None else circuit_weights[j][i]
                    batch_evals[j] += np.sum(pauli_evals * pauli_weights) * circuit_weight / shots_per_circuit
                    batch_total_probs[j] += total_meas_circuit_counts * circuit_weight / (len(measurement_pauli_dicts)*shots_per_circuit)

        batch_evals = np.array(batch_evals)
        batch_total_probs = np.array(batch_total_probs)

        # renormalize expectation values:
        if renormalize:
            batch_evals /= batch_total_probs

        return batch_evals.tolist()

    def run(self, circuits,
            observable,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):

        job_array, job_pauli_array = self.launch_jobs(
            circuits=circuits,
            observable=observable,
            observable_qubits=observable_qubits,
            conditional_qubits=conditional_qubits,
            shots_per_circuit=shots_per_circuit,
            parameters=parameters,
            result_masks=result_masks,
            circuit_weights=circuit_weights,
            renormalize=renormalize)

        job_evals = self.get_results(
            job_array=job_array,
            job_pauli_array=job_pauli_array,
            circuits=circuits,
            observable=observable,
            observable_qubits=observable_qubits,
            conditional_qubits=conditional_qubits,
            shots_per_circuit=shots_per_circuit,
            parameters=parameters,
            result_masks=result_masks,
            circuit_weights=circuit_weights,
            renormalize=renormalize
        )

        return job_evals

class KrausSeriesTomography(KrausSeriesPrimitive):

    def __init__(self, qiskit_backend, basis_subset=None, **kwargs):
        super().__init__(qiskit_backend, **kwargs)

        # ensure basis subset is valid:
        self.basis_subset = basis_subset
        if basis_subset is not None:
            for pauli_str in basis_subset:
                assert 'I' not in pauli_str
        
    def launch_jobs(self, 
            circuits,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            pure_statevector=False,
            positive_definite=True,
            renormalize=True):
        
        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))

        job_array = []
        for i, circuit in enumerate(circuits):

            # resolve any conditional qubits:
            if conditional_qubits is None:
                conditional_qubits = []
            
            circuit_weight = 1.0 if circuit_weights is None else circuit_weights[i]
            basis_generator = enumerate_pauli_strings(len(observable_qubits), 'ZXY') \
                if self.basis_subset is None else self.basis_subset

            measurement_jobs = {}

            # perform measurements over all Pauli bases:
            for basis_str in basis_generator:
                assert(len(basis_str) == len(observable_qubits))
                
                # build measurement circuit:
                measurement_qubits = observable_qubits+conditional_qubits
                meas_circuit = QuantumCircuit(len(circuit.qubits), len(measurement_qubits),
                                        name=circuit.name+f'_tomg_{basis_str.lower()}')
                meas_circuit.append(circuit.decompose(), list(range(len(circuit.qubits))))
                meas_circuit.append(make_basis_measurement_circuit(basis_str), observable_qubits)
                meas_circuit.measure(measurement_qubits,list(range(len(measurement_qubits))))

                # bind parameters, transpile, and execute circuits:
                bound_circuits = self._bind_parameters(meas_circuit, parameters, series_idx=i)
                trans_circuits = self._transpile(bound_circuits)
                job = self._launch_job(trans_circuits, shots=shots_per_circuit)
                measurement_jobs[basis_str] = job

            job_array.append(measurement_jobs)

        return job_array

    def get_results(self,
            job_array,
            circuits,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            pure_statevector=False,
            positive_definite=True,
            renormalize=True):
        
        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))
        tomography_bitmasks = np.arange(2**len(observable_qubits))

        job_density_matrices = [
            np.zeros((2**len(observable_qubits),2**len(observable_qubits)),
            dtype=np.complex128)
            for _ in range(parameters.shape[0])
        ]
        renorm_factor = 1<<len(observable_qubits)

        assert len(job_array) == len(circuits)
        for i, circuit in enumerate(circuits):

            # resolve any conditional qubits:
            if conditional_qubits is None:
                conditional_qubits = []
            
            measurement_jobs = job_array[i]
            
            # initialize maps of pauli_strings -> [counts, subbasis measurements]
            job_circuit_pauli_dicts = [{} for _ in range(len(job_density_matrices))]

            # perform measurements over all Pauli bases:
            for basis_str, job in measurement_jobs.items():
                assert(len(basis_str) == len(observable_qubits))
                
                # # build measurement circuit:
                measurement_qubits = observable_qubits+conditional_qubits
                
                job_counts = self._get_job_results(job)

                for circuit_pauli_dict, counts in zip(job_circuit_pauli_dicts, job_counts):
                    
                    # get conditional results:
                    if conditional_qubits:
                        conditional_qubit_idxs = [ measurement_qubits.index(q) for q in conditional_qubits ]
                        counts = get_conditional_result(counts, conditional_qubit_idxs)

                    # construct bitmask dictionary for each observable in this basis:
                    total_meas_circuit_counts = np.zeros_like(tomography_bitmasks)
                    for k, n in counts.items():
                        parities = (-1)**basis_measurement_parity(k,tomography_bitmasks)
                        total_meas_circuit_counts += parities*n

                    # update circuit pauli dircionary map:
                    for bitmask, x in zip(tomography_bitmasks, total_meas_circuit_counts):
                        if x != 0:
                            subbasis_str = subbasis_str_from_bitmask(bitmask, basis_str)
                            if subbasis_str not in circuit_pauli_dict:
                                circuit_pauli_dict[subbasis_str] = [0,0]
                            
                            nm_list = circuit_pauli_dict[subbasis_str]
                            nm_list[0] += x
                            nm_list[1] += 1

            for j, circuit_pauli_dict in enumerate(job_circuit_pauli_dicts):

                if not circuit_pauli_dict:
                    continue

                # convert circuit count Pauli dicts to float-weighted Pauli dicts:
                circuit_weight = 1.0 if circuit_weights is None else circuit_weights[j][i]
                for subbasis_str, nm_list in circuit_pauli_dict.items():
                    circuit_pauli_dict[subbasis_str] = (circuit_weight * nm_list[0] / (nm_list[1] * renorm_factor * shots_per_circuit))

                # Convert circuit Pauli dict to operator form:
                circuit_density = make_operator_from_pauli_dict(circuit_pauli_dict)

                if result_masks is not None:
                    mask = np.array(result_masks[i], dtype=np.float64)
                    circuit_density = mask.reshape(1,-1)*circuit_density*mask.reshape(-1,1)

                if job_density_matrices[j].shape != circuit_density.shape and not np.any(job_density_matrices[j]):
                    job_density_matrices[j] = circuit_density.astype(np.complex64)
                else:
                    job_density_matrices[j] += circuit_density

        # do density matrix postprocessing:
        for j, density_matrix in enumerate(job_density_matrices):

            # move additional density matrix axes to first two axes:
            n_additional_axes = (density_matrix.ndim-2)
            if density_matrix.ndim > 2:
                density_matrix = density_matrix.transpose(tuple(range(2,density_matrix.ndim))+(0,1))

            # if pure statevector requested, return closest pure state:
            if pure_statevector:
                _, U = np.linalg.eigh(density_matrix)
                psi = U[:,-1]
                psi /= np.sum(np.abs(psi)**2)
                job_density_matrices[j] = psi
                continue

            # force matrix to be positive definite:
            if positive_definite:
                density_eigs = np.linalg.eigvalsh(density_matrix)
                min_eigs = np.minimum(np.min(density_eigs, axis=-1),0)
                eye_matrix = np.eye(density_matrix.shape[-1]).reshape(
                    (1,)*n_additional_axes+density_matrix.shape[-2:]
                )
                density_matrix += eye_matrix*np.abs(min_eigs).reshape(min_eigs.shape+(1,1))

            # renormalize expectation values:
            if renormalize:
                trace = np.array(np.trace(density_matrix, axis1=-2,axis2=-1))
                density_matrix /= trace.reshape(trace.shape + (1,1))

            job_density_matrices[j] = density_matrix

        return job_density_matrices

    def run(self, circuits,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            pure_statevector=False,
            positive_definite=True,
            renormalize=True):
        
        job_array = self.launch_jobs(
            circuits=circuits,
            observable_qubits=observable_qubits,
            conditional_qubits=conditional_qubits,
            shots_per_circuit=shots_per_circuit,
            parameters=parameters,
            result_masks=result_masks,
            circuit_weights=circuit_weights,
            pure_statevector=pure_statevector,
            positive_definite=positive_definite,
            renormalize=renormalize)
        
        job_density_matrices = self.get_results(
            job_array=job_array, 
            circuits=circuits,
            observable_qubits=observable_qubits,
            conditional_qubits=conditional_qubits,
            shots_per_circuit=shots_per_circuit,
            parameters=parameters,
            result_masks=result_masks,
            circuit_weights=circuit_weights,
            pure_statevector=pure_statevector,
            positive_definite=positive_definite,
            renormalize=renormalize)

        return job_density_matrices

    def _run(self, circuits,
            observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            pure_statevector=False,
            positive_definite=True,
            renormalize=True):

        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))
        
        tomography_bitmasks = np.arange(2**len(observable_qubits))

        batch_density_matrices = [
            np.zeros((2**len(observable_qubits),2**len(observable_qubits)),
            dtype=np.complex128)
            for _ in range(parameters.shape[0])
        ]

        renorm_factor = 1<<len(observable_qubits)

        for i, circuit in enumerate(circuits):

            # resolve any conditional qubits:
            if conditional_qubits is None:
                conditional_qubits = []
            
            basis_generator = enumerate_pauli_strings(len(observable_qubits), 'ZXY') \
                if self.basis_subset is None else self.basis_subset

            # initialize maps of pauli_strings -> [counts, subbasis measurements]
            batch_circuit_pauli_dicts = [{} for _ in range(len(batch_density_matrices))]

            # perform measurements over all Pauli bases:
            for basis_str in basis_generator:
                assert(len(basis_str) == len(observable_qubits))
                
                # build measurement circuit:
                measurement_qubits = observable_qubits+conditional_qubits
                meas_circuit = QuantumCircuit(len(circuit.qubits), len(measurement_qubits))
                meas_circuit.append(circuit.decompose(), list(range(len(circuit.qubits))))
                meas_circuit.append(make_basis_measurement_circuit(basis_str), observable_qubits)
                meas_circuit.measure(measurement_qubits,list(range(len(measurement_qubits))))

                # bind parameters, transpile, and execute circuits:
                bound_circuits = self._bind_parameters(meas_circuit, parameters, series_idx=i)
                trans_circuits = self._transpile(bound_circuits)
                job = self._launch_job(trans_circuits, shots=shots_per_circuit)
                job_counts = self._get_job_results(job)

                for circuit_pauli_dict, counts in zip(batch_circuit_pauli_dicts, job_counts):
                    
                    # get conditional results:
                    if conditional_qubits:
                        conditional_qubit_idxs = [ measurement_qubits.index(q) for q in conditional_qubits ]
                        counts = get_conditional_result(counts, conditional_qubit_idxs)

                    # construct bitmask dictionary for each observable in this basis:
                    total_meas_circuit_counts = np.zeros_like(tomography_bitmasks)
                    for k, n in counts.items():
                        parities = (-1)**basis_measurement_parity(k,tomography_bitmasks)
                        total_meas_circuit_counts += parities*n

                    # update circuit pauli dictionary map:
                    for bitmask, x in zip(tomography_bitmasks, total_meas_circuit_counts):
                        if x != 0:
                            subbasis_str = subbasis_str_from_bitmask(bitmask, basis_str)
                            if subbasis_str not in circuit_pauli_dict:
                                circuit_pauli_dict[subbasis_str] = [0,0]
                            
                            nm_list = circuit_pauli_dict[subbasis_str]
                            nm_list[0] += x
                            nm_list[1] += 1

            for j, circuit_pauli_dict in enumerate(batch_circuit_pauli_dicts):

                if not circuit_pauli_dict:
                    continue

                # convert circuit count Pauli dicts to float-weighted Pauli dicts:
                circuit_weight = 1.0 if circuit_weights is None else circuit_weights[j][i]
                for subbasis_str, nm_list in circuit_pauli_dict.items():
                    circuit_pauli_dict[subbasis_str] = (circuit_weight * nm_list[0] / (nm_list[1] * renorm_factor * shots_per_circuit))

                # Convert circuit Pauli dict to operator form:
                circuit_density = make_operator_from_pauli_dict(circuit_pauli_dict)

                if result_masks is not None:
                    mask = np.array(result_masks[i], dtype=np.float64)
                    circuit_density = mask.reshape(1,-1)*circuit_density*mask.reshape(-1,1)

                if batch_density_matrices[j].shape != circuit_density.shape and not np.any(batch_density_matrices[j]):
                    batch_density_matrices[j] = circuit_density.astype(np.complex64)
                else:
                    batch_density_matrices[j] += circuit_density

        # do density matrix postprocessing:
        for j, density_matrix in enumerate(batch_density_matrices):

            # move additional density matrix axes to first two axes:
            n_additional_axes = (density_matrix.ndim-2)
            if density_matrix.ndim > 2:
                density_matrix = density_matrix.transpose(tuple(range(2,density_matrix.ndim))+(0,1))

            # if pure statevector requested, return closest pure state:
            if pure_statevector:
                _, U = np.linalg.eigh(density_matrix)
                psi = U[:,-1]
                psi /= np.sum(np.abs(psi)**2)
                batch_density_matrices[j] = psi
                continue

            # force matrix to be positive definite:
            if positive_definite:
                density_eigs = np.linalg.eigvalsh(density_matrix)
                min_eigs = np.minimum(np.min(density_eigs, axis=-1),0)
                eye_matrix = np.eye(density_matrix.shape[-1]).reshape(
                    (1,)*n_additional_axes+density_matrix.shape[-2:]
                )
                density_matrix += eye_matrix*np.abs(min_eigs).reshape(min_eigs.shape+(1,1))

            # renormalize expectation values:
            if renormalize:
                trace = np.array(np.trace(density_matrix, axis1=-2,axis2=-1))
                density_matrix /= trace.reshape(trace.shape + (1,1))

            batch_density_matrices[j] = density_matrix

        return batch_density_matrices

class KrausSeriesDensitySampler(KrausSeriesPrimitive):
    
    def __init__(self, qiskit_backend, **kwargs):
        super().__init__(qiskit_backend, **kwargs)
    
    def launch_jobs(self,
            circuits,
            right_observable_qubits,
            left_observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None):

        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))

        job_array = []
        for i, circuit in enumerate(circuits):

            # build measurement circuit:
            assert(len(left_observable_qubits) == len(right_observable_qubits))
            if conditional_qubits is None:
                conditional_qubits = []
            measurement_qubits = right_observable_qubits + left_observable_qubits + conditional_qubits
            meas_circuit = QuantumCircuit(len(circuit.qubits), len(measurement_qubits))
            meas_circuit.append(circuit, list(range(len(circuit.qubits))))
            meas_circuit.measure(measurement_qubits,list(range(len(measurement_qubits))))

            # bind circuit parameters and execute:
            bound_circuits = self._bind_parameters(meas_circuit, parameters, series_idx=i)
            trans_circuits = self._transpile(bound_circuits)
            job = self._launch_job(trans_circuits, shots=shots_per_circuit)
            job_array.append(job)

        return job_array

    def get_results(self,
            job_array,
            circuits,
            right_observable_qubits,
            left_observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):

        if not isinstance(circuits, Iterable):
            circuits = [circuits]
        
        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))
        job_sample_populations = [ {} for _ in range(parameters.shape[0]) ]

        for i, circuit in enumerate(circuits):
            job = job_array[i]
            job_counts = self._get_job_results(job)

            for sample_populations, counts in zip(job_sample_populations, job_counts):
                
                # get measurement qubits:
                assert(len(left_observable_qubits) == len(right_observable_qubits))
                if conditional_qubits is None:
                    conditional_qubits = []
                measurement_qubits = right_observable_qubits + left_observable_qubits + conditional_qubits

                # get conditional results:
                if conditional_qubits:
                    conditional_qubit_idxs = [ measurement_qubits.index(q) for q in conditional_qubits ]
                    counts = get_conditional_result(counts, conditional_qubit_idxs)

                # convert counts to density result dict:
                density_counts = to_density_result_dict(counts, len(left_observable_qubits))

                # mask measured counts:
                if result_masks is not None:
                    density_counts = apply_density_result_mask(density_counts, np.array(result_masks[i]))

                # add results to total probability distribution:
                weight = 1.0 if circuit_weights is None else circuit_weights[i]
                for k, n in density_counts.items():
                    if k not in sample_populations:
                        sample_populations[k] = 0.0
                    sample_populations[k] += weight*np.sqrt(n/shots_per_circuit)

        # Renormalize sample probabilities:
        if renormalize:
            for sample_populations in job_sample_populations:
                if (trace := density_trace(sample_populations)) > 0:
                    for k in sample_populations:
                        sample_populations[k] /= trace

        return job_sample_populations

    def _run(self,
            circuits,
            right_observable_qubits,
            left_observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):

        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))
        batch_sample_populations = [ {} for _ in range(parameters.shape[0]) ]
        
        # launch job array (if not given):
        if job_array is None:
            job_array = []
            for i, circuit in enumerate(circuits):

                # build measurement circuit:
                assert(len(left_observable_qubits) == len(right_observable_qubits))
                if conditional_qubits is None:
                    conditional_qubits = []
                measurement_qubits = right_observable_qubits + left_observable_qubits + conditional_qubits
                meas_circuit = QuantumCircuit(len(circuit.qubits), len(measurement_qubits))
                meas_circuit.append(circuit, list(range(len(circuit.qubits))))
                meas_circuit.measure(measurement_qubits,list(range(len(measurement_qubits))))

                # bind circuit parameters and execute:
                bound_circuits = self._bind_parameters(meas_circuit, parameters, series_idx=i)
                trans_circuits = self._transpile(bound_circuits)
                job = self._launch_job(trans_circuits, shots=shots_per_circuit)
                job_array.append(job)

        # retrieve results of job array:
        for i, circuit in enumerate(circuits):
            job = job_array[i]
            job_counts = self._get_job_results(job)

            for sample_populations, counts in zip(batch_sample_populations, job_counts):

                # get conditional results:
                if conditional_qubits:
                    conditional_qubit_idxs = [ measurement_qubits.index(q) for q in conditional_qubits ]
                    counts = get_conditional_result(counts, conditional_qubit_idxs)

                # convert counts to density result dict:
                density_counts = to_density_result_dict(counts, len(left_observable_qubits))

                # mask measured counts:
                if result_masks is not None:
                    density_counts = apply_density_result_mask(density_counts, np.array(result_masks[i]))

                # add results to total probability distribution:
                weight = 1.0 if circuit_weights is None else circuit_weights[i]
                for k, n in density_counts.items():
                    if k not in sample_populations:
                        sample_populations[k] = 0.0
                    sample_populations[k] += weight*np.sqrt(n/shots_per_circuit)

        # Renormalize sample probabilities:
        if renormalize:
            for sample_populations in batch_sample_populations:
                if (trace := density_trace(sample_populations)) > 0:
                    for k in sample_populations:
                        sample_populations[k] /= trace

        return batch_sample_populations

    def run(self,
            circuits,
            right_observable_qubits,
            left_observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):

        job_array = self.launch_jobs(
            circuits=circuits,
            right_observable_qubits=right_observable_qubits,
            left_observable_qubits=left_observable_qubits,
            conditional_qubits=conditional_qubits,
            shots_per_circuit=shots_per_circuit,
            parameters=parameters)

        job_sample_populations = self.get_results(
            job_array=job_array,
            circuits=circuits,
            right_observable_qubits=right_observable_qubits,
            left_observable_qubits=left_observable_qubits,
            shots_per_circuit=shots_per_circuit,
            parameters=parameters,
            result_masks=result_masks,
            circuit_weights=circuit_weights,
            renormalize=renormalize)

        return job_sample_populations

class KrausSeriesDensityEstimator(KrausSeriesPrimitive):

    def __init__(self, qiskit_backend, **kwargs):
        super().__init__(qiskit_backend, **kwargs)
        
    def run(self, circuits,
            observable,
            right_observable_qubits,
            left_observable_qubits,
            conditional_qubits=None,
            shots_per_circuit=1024,
            parameters=None,
            result_masks=None,
            circuit_weights=None,
            renormalize=True):

        if not isinstance(circuits, Iterable):
            circuits = [circuits]

        # format parameters to the correct size:
        parameters = self._format_parameters(parameters, len(circuits))
        batch_evals = np.zeros(parameters.shape[0])
        batch_total_probs = np.zeros(parameters.shape[0])

        # mask observables:
        if result_masks is None:
            masked_observables = [observable]*len(circuits)
        else:
            result_masks = [ np.array(m) for m in result_masks ]
            masked_observables = [
                mask.reshape(1,-1)*observable*mask.reshape(-1,1)
                for mask in result_masks
            ]

        for i, circuit in enumerate(circuits):
            
            # resolve any conditional qubits:
            if conditional_qubits is None:
                conditional_qubits = []

            # Note: if renormalizing, we still need estimates of the final probabilities
            # of the non-environment subsystem, so we use a trivial Pauli dict:
            pauli_dict = to_pauli_dict(masked_observables[i])
            if not pauli_dict:
                if renormalize:
                    pauli_dict = {'II' : 0.0}
                else:
                    continue

            # decompose observable into commuting groups of weighted Pauli strings:
            measurement_pauli_dicts = get_measurement_basis_pauli_dicts(pauli_dict)
            circuit_weight = 1.0 if circuit_weights is None else circuit_weights[i]

            for basis_str, meas_dict in measurement_pauli_dicts.items():
                
                # build measurement circuit:
                assert(len(basis_str) == len(left_observable_qubits))
                assert(len(left_observable_qubits) == len(right_observable_qubits))
                measurement_qubits = left_observable_qubits + right_observable_qubits + conditional_qubits
                meas_circuit = QuantumCircuit(len(circuit.qubits), len(measurement_qubits))
                meas_circuit.append(circuit, list(range(len(circuit.qubits))))
                meas_circuit.append(make_basis_measurement_circuit(basis_str, conj=False), left_observable_qubits)
                meas_circuit.append(make_basis_measurement_circuit(basis_str, conj=True), right_observable_qubits)
                meas_circuit.measure(measurement_qubits,list(range(len(measurement_qubits))))

                # bind parameters, transpile, and execute circuits:
                bound_circuits = self._bind_parameters(meas_circuit, parameters, series_idx=i)
                trans_circuits = self._transpile(bound_circuits)
                job = self._launch_job(trans_circuits, shots=shots_per_circuit)
                job_counts = self._get_job_results(job)

                for j, counts in enumerate(job_counts):
                    # get conditional results:
                    if conditional_qubits:
                        conditional_qubit_idxs = [ measurement_qubits.index(q) for q in conditional_qubits ]
                        counts = get_conditional_result(counts, conditional_qubit_idxs)

                    # convert counts to density result dict:
                    density_counts = to_density_result_dict(counts, len(left_observable_qubits))

                    # construct bitmask dictionary for each observable in this basis:
                    pauli_strs = list(meas_dict.keys())
                    bitmasks = np.array([ basis_bitmask(p) for p in pauli_strs ])
                    pauli_weights = np.array([ meas_dict[p] for p in pauli_strs ])
                    pauli_evals = np.zeros(len(pauli_strs))
                    
                    total_trace_probs = 0.0
                    for (row,col), n in density_counts.items():
                        if row == col:
                            parities = (-1)**basis_measurement_parity(row,bitmasks)
                            prob = np.sqrt(n/shots_per_circuit)
                            pauli_evals += parities*prob
                            total_trace_probs += prob

                    batch_evals[j] += np.sum(pauli_evals* pauli_weights) * circuit_weight
                    batch_total_probs[j] += total_trace_probs
        
        # renormalize expectation values:
        if renormalize:
            batch_evals /= batch_total_probs

        return batch_evals.tolist()
