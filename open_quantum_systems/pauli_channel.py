import numpy as np
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator
from qiskit.primitives import Estimator, BackendEstimator, BackendSampler

from .lindblad import simulate_lindblad_trajectory
from .math_util import make_pauli_operator
from .tomography import KrausSeriesEstimator, KrausSeriesSampler, KrausSeriesTomography

def make_pauli_string_circuit(n_qubits, pauli_str, conj=False):
    qc_name = f'pauli_{pauli_str.lower()}_conj' if conj \
              else f'pauli_{pauli_str.lower()}'
    qc = QuantumCircuit(n_qubits, name=qc_name)

    for i, ch in enumerate(pauli_str.upper()):
        if ch == 'X':
            qc.x(i)
        elif ch == 'Y':
            if conj:
                qc.x(i)
                qc.y(i)
                qc.x(i)
            else:
                qc.y(i)
        
        elif ch == 'Z':
            qc.z(i)
        elif ch != 'I':
            raise(ValueError(f'Unknown Pauli character: {ch}'))

    return qc

def make_pauli_string_exponential_circuit(n_qubits, pauli_str, conj=False):
    
    qc = QuantumCircuit(n_qubits+1)
    for i, ch in enumerate(pauli_str.upper()):
        if ch == 'X':
            qc.cx(-1, i)
        elif ch == 'Y':
            if conj:
                qc.x(i)
                qc.cy(-1, i)
                qc.x(i)
            else:
                qc.cy(-1, i)
        
        elif ch == 'Z':
            qc.cz(-1, i)
        elif ch != 'I':
            raise(ValueError(f'Unknown Pauli character: {ch}'))

    return qc

def make_stinespring_pauli_channel_circuits(n_qubits, pauli_strings):

    qc = QuantumCircuit(n_qubits+len(pauli_strings),
            name='pauli_stinespring')
    conj_qc = QuantumCircuit(n_qubits+len(pauli_strings),
            name='pauli_stinespring_conj')

    system_qubits = list(range(n_qubits))
    ancilla_qubits = list(range(n_qubits, len(qc.qubits)))

    thetas = ParameterVector('theta', len(pauli_strings))

    for i, str in enumerate(pauli_strings):

        # pad pauli string to size:
        assert(len(str) <= n_qubits)
        str = 'I'*(n_qubits-len(str)) + str

        # apply rotation to controlling ancilla:
        qc.ry(thetas[i], ancilla_qubits[i])
        conj_qc.ry(thetas[i], ancilla_qubits[i])

        # make pauli string exponential circuits:
        str_qc = make_pauli_string_exponential_circuit(n_qubits, str, conj=False)
        conj_str_qc = make_pauli_string_exponential_circuit(n_qubits, str, conj=True)

        # add the controlled exponential circuits:
        qc.append(str_qc, system_qubits+[ancilla_qubits[i]])
        conj_qc.append(str_qc, system_qubits+[ancilla_qubits[i]])

    return qc, conj_qc, thetas

def get_string_product_idxs(gammas, order):

    # obtain string indexes in order of increasing gamma:
    string_idxs = np.argsort(np.array(gammas))

    # take increasing number of combinations from idxs:
    string_product_idxs = [
        idxs
        for n in range(min(order+1,len(gammas)+1))
        for idxs in combinations(string_idxs, n)
    ]
    
    return string_product_idxs

def get_stinespring_pauli_channel_timeseries_parameters(t_vals, gammas):
    
    gammas_np = np.array(gammas).reshape(1,-1)
    t_np = np.array(t_vals.reshape(-1,1))
    
    # compute the probability of applying each Pauli string
    # (this arises directly from the Kraus series):
    flip_probs = (1+np.exp(-2*gammas_np*t_np))/2
    thetas = 2*np.arccos(np.sqrt(flip_probs))
    thetas = thetas.reshape(thetas.shape[0], 1, thetas.shape[1])

    return thetas

class PauliChannelSimulation:

    def __init__(self,
                 n_qubits=2,
                 pauli_strings=['XX'],
                 gammas=1,
                 kraus_order=2):
        super().__init__()

        # perform sanity check:
        assert len(gammas) == len(pauli_strings), "Gammas must have same length as pauli strings."
        for str in pauli_strings:
            assert len(str) == n_qubits, "Pauli string length must equal number of qubits."

        # initialize variables:
        self.n_qubits = n_qubits
        self.pauli_strings = pauli_strings
        self.gammas = [gammas]*len(self.pauli_strings) if np.isscalar(gammas) else gammas
        self.kraus_order = kraus_order

        # construct list of Pauli string circuits:
        self.pauli_string_circuits = [ make_pauli_string_circuit(n_qubits, str) for str in pauli_strings ]
        self.conj_pauli_string_circuits = [ make_pauli_string_circuit(n_qubits, str, conj=True) for str in pauli_strings ]

        # construct full density circuit:
        self.stinespring_circuit, self.conj_stinespring_circuit, self.thetas = \
            make_stinespring_pauli_channel_circuits(n_qubits, pauli_strings)

    def energies(self):
        H_diag = np.zeros(2**self.n_qubits)
        return H_diag

    def hamiltonian_operator(self):
        return np.diag(self.energies())

    def lindblad_operators(self):
        return [
            make_pauli_operator(str) for str in self.pauli_strings
        ]

    def linblad_hermitian_operator(self):
        return np.array(self.lindblad_operators()).sum(0)

    def _run_pure_kraus_circuit_sampler(self,
                                        circuits,
                                        t_params,
                                        circuit_weights,
                                        backend,
                                        shots,
                                        renormalize=True,
                                        submit_jobs_only=False,
                                        submitted_jobs_data=None,
                                        **sampler_kwargs):
        
        sampler = KrausSeriesSampler(backend, **sampler_kwargs)

        # launch jobs:
        if submitted_jobs_data is None:
            submitted_jobs_data = sampler.launch_jobs(
                circuits,
                observable_qubits=list(range(self.n_qubits)),
                conditional_qubits=None,
                shots_per_circuit=shots,
                parameters=t_params,
                result_masks=None,
                circuit_weights=circuit_weights,
                renormalize=renormalize)

        # return job data if requested:
        if submit_jobs_only:
            return submitted_jobs_data

        # process job results:
        job_array = submitted_jobs_data
        sampler_results = sampler.get_results(
                            job_array,
                            circuits,
                            observable_qubits=list(range(self.n_qubits)),
                            conditional_qubits=None,
                            shots_per_circuit=shots,
                            parameters=t_params,
                            result_masks=None,
                            circuit_weights=circuit_weights,
                            renormalize=renormalize)

        if circuit_weights is None:
            sampler_probs = np.zeros((len(sampler_results),2**self.n_qubits))
            for i, prob_dict in enumerate(sampler_results):
                for j, p in prob_dict.items():
                    sampler_probs[i,j] = p
        else:
            assert(len(sampler_results) == 1)
            sampler_probs = np.zeros((len(circuit_weights[0][0]), 2**self.n_qubits))
            for j, p in sampler_results[0].items():
                sampler_probs[:,j] = p

        return sampler_probs


    def _run_pure_kraus_circuit_estimator(self,
                                          circuits,
                                          t_params,
                                          circuit_weights,
                                          backend,
                                          observable,
                                          shots,
                                          renormalize=True,
                                          submit_jobs_only=False,
                                          submitted_jobs_data=None,
                                          **estimator_kwargs):
        
        estimator = KrausSeriesEstimator(backend, **estimator_kwargs)

        # launch jobs:
        if submitted_jobs_data is None:
            submitted_jobs_data = estimator.launch_jobs(
                circuits,
                observable,
                observable_qubits=list(range(self.n_qubits)),
                conditional_qubits=None,
                shots_per_circuit=shots,
                parameters=t_params,
                result_masks=None,
                circuit_weights=circuit_weights,
                renormalize=renormalize)
        
        # return job data if requested:
        if submit_jobs_only:
            return submitted_jobs_data

        # process_job_results:
        job_array, job_pauli_array = submitted_jobs_data
        eval_results = np.real_if_close(estimator.get_results(
                    job_array,
                    job_pauli_array,
                    circuits,
                    observable,
                    observable_qubits=list(range(self.n_qubits)),
                    conditional_qubits=None,
                    shots_per_circuit=shots,
                    parameters=t_params,
                    result_masks=None,
                    circuit_weights=circuit_weights,
                    renormalize=renormalize))

        if circuit_weights is not None:
            eval_results = eval_results.flatten()

        return eval_results

    def _run_pure_kraus_series_tomography(self,
                                          circuits,
                                          t_params,
                                          circuit_weights,
                                          backend,
                                          shots,
                                          renormalize=True,
                                          positive_definite=True,
                                          submit_jobs_only=False,
                                          submitted_jobs_data=None,
                                          **sampler_kwargs):
        
        tomography = KrausSeriesTomography(backend, **sampler_kwargs)

        # launch jobs:
        if submitted_jobs_data is None:
            submitted_jobs_data = tomography.launch_jobs(
                circuits,
                observable_qubits=list(range(self.n_qubits)),
                conditional_qubits=None,
                shots_per_circuit=shots,
                parameters=t_params,
                result_masks=None,
                circuit_weights=circuit_weights,
                renormalize=renormalize,
                positive_definite=positive_definite)
        
        # return job data if requested:
        if submit_jobs_only:
            return submitted_jobs_data

        # process job results:
        job_array = submitted_jobs_data
        tomography_results = tomography.get_results(
            job_array,
            circuits,
            observable_qubits=list(range(self.n_qubits)),
            conditional_qubits=None,
            shots_per_circuit=shots,
            parameters=t_params,
            result_masks=None,
            circuit_weights=circuit_weights,
            renormalize=renormalize,
            positive_definite=positive_definite)

        return tomography_results
    
    def simulate_pure_state_evolution(self, 
                                      pure_state,
                                      t,
                                      backend,
                                      observable=None,
                                      shots=1024,
                                      renormalize=True,
                                      transpile_options={},
                                      run_options={},
                                      result_options={},
                                      primitive_kwargs={}):

        # generate state preparation gate:
        pure_state = np.array(pure_state).squeeze()
        state_prep_gate = StatePreparation(pure_state)

        # enumerate Pauli string indexes in order of decreasing effectiveness:
        string_idxs = get_string_product_idxs(
            self.gammas, 
            order=self.kraus_order)
        
        pure_state_kraus_circuits = []
        kraus_circuit_weights = []

        # compute probabilities of each flip happening independently:
        gammas_np = np.array(self.gammas).reshape(1,-1)
        gamma_noflip_probs = (1+np.exp(-2*gammas_np*t.reshape(-1,1)))/2

        for idxs in string_idxs:
            
            # compute the weight of each circuit, which is the
            # probability of a sequence of flips (and non-flips) occurring:
            _idxs = np.array(idxs)
            gamma_flip_idx_probs = gamma_noflip_probs.copy()
            if len(_idxs) > 0:
                gamma_flip_idx_probs[:,_idxs] = 1-gamma_flip_idx_probs[:,_idxs]
            flip_probs = np.prod(gamma_flip_idx_probs, axis=-1)
            kraus_circuit_weights.append(flip_probs)
            
            # construct circuit for sequence of flips:
            idx_strings = '_'.join([ self.pauli_strings[i] for i in idxs ])
            pure_qc = QuantumCircuit(self.n_qubits, name=f'pauli_{idx_strings}')
            all_qubits = list(range(self.n_qubits))
            pure_qc.append(state_prep_gate, all_qubits)
            for i in idxs:
                pure_qc.append(self.pauli_string_circuits[i], all_qubits)
            
            pure_state_kraus_circuits.append(pure_qc)

        primitive_kwargs |= {
            'qiskit_transpile_options' : transpile_options,
            'qiskit_run_options' : run_options,
            'qiskit_result_options' : result_options,
        }

        # simulate circuits for each order term in the Kraus series:
        if observable is None:
            return self._run_pure_kraus_circuit_sampler(
                circuits=pure_state_kraus_circuits,
                t_params=None,
                circuit_weights=[kraus_circuit_weights],
                backend=backend,
                shots=shots,
                renormalize=renormalize,
                **primitive_kwargs)
        else:
            return self._run_pure_kraus_circuit_estimator(
                circuits=pure_state_kraus_circuits,
                t_params=None,
                circuit_weights=[kraus_circuit_weights],
                backend=backend,
                observable=observable,
                shots=shots,
                renormalize=renormalize,
                **primitive_kwargs)


    def simulate_stinespring_pure_state_evolution(self,
                                      pure_state,
                                      t,
                                      backend,
                                      observable=None,
                                      shots=1024,
                                      renormalize=True,
                                      transpile_options={},
                                      run_options={},
                                      result_options={},
                                      primitive_kwargs={}):
        
        # generate state preparation gate:
        pure_state = np.array(pure_state).squeeze()

        qc = QuantumCircuit(len(self.stinespring_circuit.qubits),
                                name=self.stinespring_circuit.name)
        state_prep_gate = StatePreparation(pure_state)
        qc.append(state_prep_gate, list(range(self.n_qubits)))
        qc.append(self.stinespring_circuit,
            list(range(len(self.stinespring_circuit.qubits))))

        # get timeseries parameters:
        t_params = get_stinespring_pauli_channel_timeseries_parameters(t,self.gammas)
        
        primitive_kwargs |= {
            'qiskit_transpile_options' : transpile_options,
            'qiskit_run_options' : run_options,
            'qiskit_result_options' : result_options,
        }

        # simulate stinespring circuit:
        if observable is None:
            return self._run_pure_kraus_circuit_sampler(
                circuits=[qc],
                t_params=t_params,
                circuit_weights=None,
                backend=backend,
                shots=shots,
                renormalize=renormalize,
                **primitive_kwargs)
        else:
            return self._run_pure_kraus_circuit_estimator(
                circuits=[qc],
                t_params=t_params,
                circuit_weights=None,
                backend=backend,
                observable=observable,
                shots=shots,
                renormalize=renormalize,
                **primitive_kwargs)

    
    def simulate_pure_state_density_evolution(self,
        pure_state,
        t,
        backend,
        shots=1024,
        renormalize=True,
        positive_definite=True,
        transpile_options={},
        run_options={},
        result_options={},
        primitive_kwargs={}):

       # generate state preparation gate:
        pure_state = np.array(pure_state).squeeze()
        state_prep_gate = StatePreparation(pure_state)

        # enumerate Pauli string indexes in order of decreasing effectiveness:
        string_idxs = get_string_product_idxs(
            self.gammas, 
            order=self.kraus_order)
        
        pure_state_kraus_circuits = []
        kraus_circuit_weights = []

        # compute probabilities of each flip happening independently:
        gammas_np = np.array(self.gammas).reshape(1,-1)
        gamma_noflip_probs = (1+np.exp(-2*gammas_np*t.reshape(-1,1)))/2

        for idxs in string_idxs:

            # compute the weight of each circuit, which is the
            # probability of a sequence of flips (and non-flips) occurring:
            _idxs = np.array(idxs)
            gamma_flip_idx_probs = gamma_noflip_probs.copy()
            if len(_idxs) > 0:
                gamma_flip_idx_probs[:,_idxs] = 1-gamma_flip_idx_probs[:,_idxs]
            flip_probs = np.prod(gamma_flip_idx_probs, axis=-1)
            kraus_circuit_weights.append(flip_probs)

            # construct circuit for sequence of flips:
            pure_qc = QuantumCircuit(self.n_qubits)
            all_qubits = list(range(self.n_qubits))
            pure_qc.append(state_prep_gate, all_qubits)
            for i in idxs:
                pure_qc.append(self.pauli_string_circuits[i], all_qubits)
            
            pure_state_kraus_circuits.append(pure_qc)

        primitive_kwargs |= {
            'qiskit_transpile_options' : transpile_options,
            'qiskit_run_options' : run_options,
            'qiskit_result_options' : result_options,
        }

        # simulate circuits for each order term in the Kraus series:
        return self._run_pure_kraus_series_tomography(
            circuits=pure_state_kraus_circuits,
            t_params=None,
            circuit_weights=kraus_circuit_weights,
            backend=backend,
            shots=shots,
            renormalize=renormalize,
            positive_definite=positive_definite,
            **primitive_kwargs).squeeze(0)

    def simulate_stinespring_pure_state_density_evolution(self,
                                      pure_state,
                                      t,
                                      backend,
                                      shots=1024,
                                      renormalize=True,
                                      positive_definite=True,
                                      transpile_options={},
                                      run_options={},
                                      result_options={},
                                      primitive_kwargs={}):
        
        # generate state preparation gate:
        pure_state = np.array(pure_state).squeeze()

        qc = QuantumCircuit(len(self.stinespring_circuit.qubits))
        state_prep_gate = StatePreparation(pure_state)
        qc.append(state_prep_gate, list(range(self.n_qubits)))
        qc.append(self.stinespring_circuit,
            list(range(len(self.stinespring_circuit.qubits))))

        # get timeseries parameters:
        t_params = get_stinespring_pauli_channel_timeseries_parameters(t,self.gammas)
        
        primitive_kwargs |= {
            'qiskit_transpile_options' : transpile_options,
            'qiskit_run_options' : run_options,
            'qiskit_result_options' : result_options,
        }

        # simulate stinespring circuit:
        return self._run_pure_kraus_series_tomography(
            circuits=[qc],
            t_params=t_params,
            circuit_weights=None,
            backend=backend,
            shots=shots,
            renormalize=renormalize,
            positive_definite=positive_definite,
            **primitive_kwargs)
        
    
    def simulate_classical_density_matrix_evolution(self,
                                                    initial_state,
                                                    t,
                                                    observable=None):
        
        rhos = simulate_lindblad_trajectory(
            initial_state, t,
            hamiltonian=self.hamiltonian_operator(),
            lindblad_ops=self.lindblad_operators(),
            gammas=self.gammas
        )

        if observable is not None:
            return np.real(np.trace(observable @ rhos, axis1=1,axis2=2))

        return rhos
    
