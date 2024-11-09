import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import XGate, RYGate, GlobalPhaseGate
from qiskit.circuit.library import StatePreparation

from open_quantum_systems.qho import *
from open_quantum_systems.math_util import *
from open_quantum_systems.lindblad import simulate_lindblad_trajectory
from open_quantum_systems.tomography import KrausSeriesEstimator, KrausSeriesSampler, KrausSeriesTomography

from scipy.special import factorial
from scipy.linalg import expm


def make_ctrl_add_n_gate(n_qubits, n):

    qc = QuantumCircuit(n_qubits+1, name=f'Ctrl-Add({n})')
    for i in range(n_qubits):
        if (n & (1<<i)):
            inc_i = make_increment_gate(n_qubits-i, control=True)
            qc.append(inc_i, [0]+list(range(i+1,n_qubits+1)))
    return qc.decompose().to_instruction()

def make_ctrl_sub_n_gate(n_qubits, n):

    qc = QuantumCircuit(n_qubits, name=f'Ctrl-Sub({n})')
    for i in range(n_qubits):
        if (n & (1<<i)):
            dec_i = make_decrement_gate(n_qubits-i, control=True)
            qc.append(dec_i, [0]+list(range(i+1,n_qubits+1)))
    return qc.decompose().to_instruction()

def make_ctrl_ladder_dilation_circuit(n_qubits, order=1, superdiag=None, thetas=None, transpose=False, theta_tolerance=1e-6):

    if superdiag is None:                    
        # create and pad thetas to correct size:
        if thetas is None:
            thetas =  ParameterVector('theta',2**n_qubits)
        if len(thetas) < 2**n_qubits:
            assert(isinstance(thetas, list))
            thetas += [0]*(2**n_qubits - len(thetas))
    else:
        superdiag = np.array(superdiag)
        
        # compute controlled rotation angles:
        betas = np.zeros(2**n_qubits, dtype=superdiag.dtype)
        betas[:superdiag.shape[0]] = superdiag
        thetas = make_inv_theta_transform(n_qubits) @ (-2*np.arcsin(betas))
        thetas[np.abs(thetas) < np.abs(theta_tolerance)] = 0.0

    qc = QuantumCircuit(n_qubits+2, name=r'Ctrl-$L_n$ Decoherence')
    
    # add addition/subtraction gate:
    if order > 0:
        if transpose:
            ctrl_add_n = make_ctrl_add_n_gate(n_qubits, order)
            qc.append(ctrl_add_n, list(range(n_qubits+1)))
        else:
            ctrl_sub_n = make_ctrl_sub_n_gate(n_qubits, order)
            qc.append(ctrl_sub_n, list(range(n_qubits+1)))

def get_J_z_unitary_transform(n_qubits, gamma_xyz=[0,0,1]):

    a = qho_ladder(n_qubits)
    a_dag = a.conj().T
    N = a_dag @ a

    J_x = (np.kron(a_dag, a) + np.kron(a, a_dag))/(2.)
    J_y = (np.kron(a_dag, a) - np.kron(a, a_dag))/(2.j)
    J_z = kron_sum([N, -N])/2.

    n = np.array(gamma_xyz, dtype=np.float64)
    gamma_norm = np.linalg.norm(gamma_xyz)
    if gamma_norm <= 0:
        return np.eye(2**(2*n_qubits))
    else:
        n /= gamma_norm

    v = (n + np.array([0,0,1]))
    v /= np.linalg.norm(v)

    U_v = expm(-1.j*np.pi*(v[0]*J_x + v[1]*J_y + v[2]*J_z))

    return U_v

def make_parameterized_diagonal_schwinger_circuits(n_qubits,
                                                   state_prep_circuit=None,
                                                   conj_state_prep_circuit=None,
                                                   kraus_order=2):

    circuits = []
    conj_circuits = []
    theta_ab_params = ParameterVector('theta_ab', 2**(2*n_qubits))
    phi_params = ParameterVector('phi', 2**n_qubits)

    for order in range(kraus_order+1):

        # create circuit:
        n_prep_qubits = 0 if state_prep_circuit is None else len(state_prep_circuit.qubits)
        order_qc = QuantumCircuit(max(2*n_qubits+1,n_prep_qubits+1))
        conj_order_qc = QuantumCircuit(max(2*n_qubits+1,n_prep_qubits+1))

        # append state preparation circuit:
        if state_prep_circuit is not None:
            assert conj_state_prep_circuit is not None, "conj_state_prep_circuit must be supplied along with state_prep_circuit."
            prep_ancillas = [i+1 for i in range(2*n_qubits, n_prep_qubits)]
            prep_qubits = list(range(n_qubits)) + prep_ancillas
            order_qc.append(state_prep_circuit, prep_qubits)
            conj_order_qc.append(conj_state_prep_circuit, prep_qubits)

        # apply dissipative operator:
        contraction_circuit = make_diagonal_contraction_circuit(2*n_qubits, thetas=theta_ab_params)
        order_qc.append(contraction_circuit, list(range(2*n_qubits+1)))
        conj_order_qc.append(contraction_circuit, list(range(2*n_qubits+1)))

        # evolve dissipated state unitarily:
        hamiltonian_circuit = make_diagonal_hamiltonian_circuit(n_qubits, phis=phi_params)
        conj_hamiltonian_circuit = make_diagonal_hamiltonian_circuit(n_qubits, phis=phi_params, conj=True)
        order_qc.append(hamiltonian_circuit, list(range(n_qubits)))
        order_qc.append(hamiltonian_circuit, list(range(n_qubits,2*n_qubits)))
        conj_order_qc.append(conj_hamiltonian_circuit, list(range(n_qubits)))
        conj_order_qc.append(conj_hamiltonian_circuit, list(range(n_qubits,2*n_qubits)))

        circuits.append(order_qc)
        conj_circuits.append(conj_order_qc)


    return circuits, conj_circuits, theta_ab_params, phi_params

def get_diagonal_schwinger_timeseries_parameters(n_qubits, 
        t_vals, 
        theta_params,
        phi_params,
        omega=1.0,
        gamma_z=1.0,
        hbar=1.0,
        kraus_order=2,
        angle_tolerance=1e-10):

    a = np.kron(np.arange(2**n_qubits), np.ones(2**n_qubits))
    b = np.kron(np.ones(2**n_qubits), np.arange(2**n_qubits))
    J_z_diag = (a - b)/2.0
    H_a_diag = hbar*omega*(0.5 + np.arange(2**n_qubits))
    parameters = []
    weights = []

    for t in t_vals:
        # compute phis (Hamiltonian evolution phase parameters):
        phis = make_inv_theta_transform(n_qubits) @ standard_angles(-t*H_a_diag/hbar)
        phis[np.abs(phis) < np.abs(angle_tolerance)] = 0
        phi_values = {phi_params[i]: phis[i] for i in range(2**n_qubits)}

        # compute theta parameters (these encode the action of the diagonal Jz operator):
        parameter_values = []
        parameter_weights = []
        for order in range(kraus_order+1):
            
            betas = np.exp(-0.5*gamma_z*t*(J_z_diag**2))
            betas *= np.sqrt((gamma_z*t)**order / factorial(order)) * (J_z_diag**order)
            beta_weight = 1.0 # np.max(np.abs(betas))

            #if beta_weight > 1.0:
            #    betas /= beta_weight

            if np.max(np.abs(betas)) > 1:
                raise ValueError(f'Order {order} Kraus operator has norm exceeding one; check t values and operator superdiagonal for invalid (i.e. negative or complex) values')

            thetas = make_inv_theta_transform(2*n_qubits) @ (-2*np.arcsin(betas))
            thetas[np.abs(thetas) < np.abs(angle_tolerance)] = 0.0
            theta_values = {theta_params[i] : thetas[i] for i in range(2**(2*n_qubits))}

            parameter_values.append(phi_values | theta_values)
            parameter_weights.append(beta_weight)

        parameters.append(parameter_values)
        weights.append(parameter_weights)

    return parameters, weights

def make_parameterized_schwinger_circuits(n_qubits,
                                          state_prep_circuit=None,
                                          conj_state_prep_circuit=None,
                                          kraus_order=2):
    circuits = []
    conj_circuits = []
    theta_ab_params = ParameterVector('theta_ab', 2*(2**n_qubits))
    phi_params = ParameterVector('phi', 2**n_qubits)

    for order in range(kraus_order+1):

        # create circuit:
        n_prep_qubits = 0 if state_prep_circuit is None else len(state_prep_circuit.qubits)
        order_qc = QuantumCircuit(max(2*n_qubits+2,n_prep_qubits+2))
        conj_order_qc = QuantumCircuit(max(2*n_qubits+2,n_prep_qubits+2))

        # append state preparation circuit:
        if state_prep_circuit is not None:
            assert conj_state_prep_circuit is not None, "conj_state_prep_circuit must be supplied along with state_prep_circuit."
            prep_ancillas = [i+2 for i in range(2*n_qubits, n_prep_qubits)]
            prep_qubits = list(range(n_qubits)) + prep_ancillas
            order_qc.append(state_prep_circuit, prep_qubits)
            conj_order_qc.append(conj_state_prep_circuit, prep_qubits)

        #TODO

        # apply dissipative operator:
        

        # evolve dissipated state unitarily:



class SchwingerOscillatorSimulation:

    def __init__(self,
                n_qubits=2,
                kraus_order=2,
                hbar=1.0,
                omega=1.0,
                gamma_xyz=[0.0, 0.0, 1.0],
                simultaneous=True,
                angle_tolerance=1e-10,
                ):
            """Constructs a simulation of a Schwinger 2D oscillator with angular damping

            Args:
                n_qubits (int, optional): Number of qubits used to represent each dimension of the 2D oscillator. Defaults to 2.
                kraus_order (int, optional): Order of the Kraus series used. For best results is recommended that the order 
                                            equals the number of the highest occupied state, Defaults to 2.
                hbar (float, optional): Value of "hbar", the reduced Planck constant. Defaults to 1.0.
                omega (float, optional): Oscillator angular frequency. Defaults to 1.0.
                gamma (float, optional): Value of the Lindblad coupling constant for the QHO annilation operator. Defaults to 1.0.
                angle_tolerance (float, optional): Tolerance for rotation gate error when compiling quantum circuits. Defaults to 1e-10.
            """

            self.n_qubits = n_qubits
            self.kraus_order = kraus_order
            self.hbar = hbar
            self.omega = omega
            self.gamma_xyz = np.array(gamma_xyz)
            self.simultaneous = simultaneous
            self.angle_tolerance = angle_tolerance

            if simultaneous:
                self.kraus_circuits, \
                self.conj_kraus_circuits, \
                self.theta_params, \
                self.phi_params = \
                    make_parameterized_diagonal_schwinger_circuits(n_qubits=n_qubits, kraus_order=kraus_order)

                self.unitary_transform = get_J_z_unitary_transform(n_qubits, gamma_xyz)
            else:
                raise NotImplementedError('Non-simultaneous form of this system is not yet supported.')

    def energies(self):
        N_ab = np.kron(np.arange(2**self.n_qubits), np.ones(2**self.n_qubits)) + \
                     np.kron(np.ones(2**self.n_qubits), np.arange(2**self.n_qubits))
        return self.hbar*self.omega*(1 + N_ab)

    def hamiltonian_operator(self):
        return np.diag(self.energies())

    def ladder_operators(self):
        a = qho_ladder(n_qubits=self.n_qubits)
        return np.kron(a,np.eye(a.shape[-1])), np.kron(np.eye(a.shape[-1]),a)

    def position_operators(self, m=1.0, unitless=False):
        a = qho_ladder(self.n_qubits)
        x0_scale = np.sqrt(self.hbar / (m*self.omega))
        x = np.sqrt((self.hbar)/(2*m*self.omega))*(a.conj().T + a)
        X, Y = np.kron(x,np.eye(x.shape[-1])), np.kron(np.eye(x.shape[-1]),x)

        if unitless:
            X /= x0_scale
            Y /= x0_scale

        return X, Y

    def momentum_operators(self, m=1.0, unitless=False):
        a = qho_ladder(self.n_qubits)
        p0_scale = np.sqrt(self.hbar*m*self.omega)
        p = 1.j*np.sqrt((self.hbar*m*self.omega)/2)*(a.conj().T - a)
        Px, Py = np.kron(p,np.eye(p.shape[-1])), np.kron(np.eye(p.shape[-1]),p)

        if unitless:
            Px /= p0_scale
            Py /= p0_scale

        return Px, Py

    def J_minus_operator(self):
        a = qho_ladder(n_qubits=self.n_qubits)
        return np.kron(a,a.conj().T)

    def J_plus_operator(self):
        a = qho_ladder(n_qubits=self.n_qubits)
        return np.kron(a.conj().T,a)

    def J_z_operator(self):
        J_z_N = np.kron(np.arange(2**self.n_qubits), np.ones(2**self.n_qubits)) - \
                     np.kron(np.ones(2**self.n_qubits), np.arange(2**self.n_qubits))
        return np.diag(J_z_N/2.0)

    def J_x_operator(self):
        a = qho_ladder(n_qubits=self.n_qubits)
        J_x = (np.kron(a.conj().T,a) + np.kron(a,a.conj().T))/2.0
        return J_x

    def J_y_operator(self):
        a = qho_ladder(n_qubits=self.n_qubits)
        J_y = (np.kron(a.conj().T,a) - np.kron(a,a.conj().T))/2.0j
        return J_y

    def J_squared_operator(self):
        N_ab = np.kron(np.arange(2**self.n_qubits), np.ones(2**self.n_qubits)) + \
                     np.kron(np.ones(2**self.n_qubits), np.arange(2**self.n_qubits))

        return np.diag((N_ab/2)*(N_ab/2 + 1))

    def _run_pure_kraus_circuit_estimator(self,
                                          circuits,
                                          t_params,
                                          backend,
                                          observable,
                                          shots,
                                          circuit_weights,
                                          renormalize=True,
                                          **estimator_kwargs
                                          ):
        estimator = KrausSeriesEstimator(backend, **estimator_kwargs)
        eval_results = np.real_if_close(estimator.run(
                    circuits,
                    observable,
                    observable_qubits=list(range(2*self.n_qubits)),
                    conditional_qubits=[2*self.n_qubits],
                    shots_per_circuit=shots,
                    parameters=t_params,
                    result_masks=None,
                    circuit_weights=circuit_weights,
                    renormalize=renormalize))

        if circuit_weights is not None:
            eval_results = eval_results.flatten()

        return eval_results

    def _run_pure_kraus_circuit_sampler(self,
                                        circuits,
                                        t_params,
                                        backend,
                                        shots,
                                        circuit_weights,
                                        renormalize=True,
                                        **sampler_kwargs):
        
        sampler = KrausSeriesSampler(backend, **sampler_kwargs)

        sampler_results = sampler.run(
                            circuits,
                            observable_qubits=list(range(2*self.n_qubits)),
                            conditional_qubits=[2*self.n_qubits],
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
            sampler_probs = np.zeros((len(circuit_weights), 2**(2*self.n_qubits)))
            for i, sampler_result in enumerate(sampler_results):
                for j, p in sampler_result.items():
                    sampler_probs[i,j] = p

        return sampler_probs

    def _run_pure_kraus_series_tomography(self,
                                          circuits,
                                          t_params,
                                          backend,
                                          shots,
                                          circuit_weights,
                                          renormalize=True,
                                          positive_definite=True,
                                          submit_jobs_only=False,
                                          submitted_jobs_data=None,
                                          **tomography_kwargs):

        tomography = KrausSeriesTomography(backend, **tomography_kwargs)
        
        # launch jobs:
        if submitted_jobs_data is None:
            submitted_jobs_data = tomography.launch_jobs(
                circuits,
                observable_qubits=list(range(2*self.n_qubits)),
                conditional_qubits=[2*self.n_qubits],
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
            observable_qubits=list(range(2*self.n_qubits)),
            conditional_qubits=[2*self.n_qubits],
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
        
        """Simulates the trajectory evolution of the system initialized in a pure state.

        Args:
            pure_state (np.ndarray): Initial pure state of the system (in the energy basis). Must be of length 2^(n_qubits).
            t (np.ndarray): Trajectory time values to sample during evolution.
            backend (qiskit.providers.Backend): Backend to use for the simulation.
            observable (np.ndarray, optional): If not None, the expectation value of this observable is returned
                                               for each t value in the trajectory. Defaults to None.
            shots (int, optional): Number of shots to use per quantum circuit. Defaults to 1024.
            renormalize (bool, optional): If True, renormalizes the total wave function obtained from the Kraus operators. Defaults to True.
            transpile_options (dict, optional): Arguments passed to qiskit's transpile() function. Defaults to {}.
            run_options (dict, optional): Arguments passed to the qiskit's backend.run() function. Defaults to {}.
            result_options (dict, optional): Arguments passed to qiskit's job.result() function. Defaults to {}.
            primitive_kwargs (dict, optional): Additional arguments passed to the KrausSeriesPrimitive object created to
                                               execute quantum circuit jobs. Defaults to {}

        Returns:
            np.ndarray: Trajectory of eigenstate occupation probabilities (or observable if provided).
        """

        # calculate circuit parameters for each timestep:
        if self.simultaneous:
            t_params, t_weights = get_diagonal_schwinger_timeseries_parameters(
                n_qubits=self.n_qubits,
                t_vals=t,
                theta_params=self.theta_params,
                phi_params=self.phi_params,
                omega=self.omega,
                gamma_z=np.linalg.norm(self.gamma_xyz),
                hbar=self.hbar,
                kraus_order=self.kraus_order,
            )
        else:
            raise NotImplementedError()

        # convert pure state to numpy array:
        pure_state = np.array(pure_state).squeeze()

        if self.simultaneous:
            pure_state = self.unitary_transform @ pure_state

            if observable is not None:
                observable = self.unitary_transform @ observable @ self.unitary_transform.conj().T

        # prepend state preparation circuit:
        pure_state_kraus_circuits = []
        for kraus_circuit in self.kraus_circuits:
            pure_qc = QuantumCircuit(len(kraus_circuit.qubits))
            state_prep_gate = StatePreparation(pure_state)
            pure_qc.append(state_prep_gate, list(range(2*self.n_qubits)))
            pure_qc.append(kraus_circuit, list(range(len(kraus_circuit.qubits))))
            pure_state_kraus_circuits.append(pure_qc.decompose())
        

        primitive_kwargs |= {
            'qiskit_transpile_options' : transpile_options,
            'qiskit_run_options' : run_options,
            'qiskit_result_options' : result_options,
        }

        # simulate circuits for each order term in the Kraus series:
        if observable is None:
            return self._run_pure_kraus_circuit_sampler(
                pure_state_kraus_circuits,
                t_params,
                backend,
                shots,
                circuit_weights=t_weights,
                renormalize=renormalize,
                **primitive_kwargs)
        else:
            return self._run_pure_kraus_circuit_estimator(
                pure_state_kraus_circuits,
                t_params,
                backend,
                observable,
                shots,
                circuit_weights=t_weights,
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
        
        """Simulates the density matrix trajectory evolution of the system initialized in a pure state.

        Args:
            pure_state (np.ndarray): Initial pure state of the system (in the energy basis, E_x times E_y). Must be of length 4^(n_qubits).
            t (np.ndarray): Trajectory time values to sample during evolution.
            backend (qiskit.providers.Backend): Backend to use for the simulation.
            shots (int, optional): Number of shots to use per quantum circuit. Defaults to 1024.
            renormalize (bool, optional): If True, renormalizes the density matrices to have a trace of one. Defaults to True.
            positive_definite (bool, optional): If True, forces reconstructed matrices to be positive (semi-)definite. Defaults to True
            transpile_options (dict, optional): Arguments passed to qiskit's transpile() function. Defaults to {}.
            run_options (dict, optional): Arguments passed to the qiskit's backend.run() function. Defaults to {}.
            result_options (dict, optional): Arguments passed to qiskit's job.result() function. Defaults to {}.
            primitive_kwargs (dict, optional): Additional arguments passed to the KrausSeriesPrimitive object created to
                                               execute quantum circuit jobs. Defaults to {}

        Returns:
            np.ndarray: Trajectory of density matrices.
        """

        # calculate circuit parameters for each timestep:
        if self.simultaneous:
            t_params, t_weights = get_diagonal_schwinger_timeseries_parameters(
                n_qubits=self.n_qubits,
                t_vals=t,
                theta_params=self.theta_params,
                phi_params=self.phi_params,
                omega=self.omega,
                gamma_z=np.linalg.norm(self.gamma_xyz),
                hbar=self.hbar,
                kraus_order=self.kraus_order,
            )
        else:
            raise NotImplementedError()

        # convert pure state to numpy array:
        pure_state = np.array(pure_state).squeeze()

        if self.simultaneous:
            pure_state = self.unitary_transform @ pure_state

        # prepend state preparation circuit:
        pure_state_kraus_circuits = []
        for kraus_circuit in self.kraus_circuits:
            pure_qc = QuantumCircuit(len(kraus_circuit.qubits))
            state_prep_gate = StatePreparation(pure_state)
            pure_qc.append(state_prep_gate, list(range(2*self.n_qubits)))
            pure_qc.append(kraus_circuit, list(range(len(kraus_circuit.qubits))))
            pure_state_kraus_circuits.append(pure_qc.decompose())

        primitive_kwargs |= {
            'qiskit_transpile_options' : transpile_options,
            'qiskit_run_options' : run_options,
            'qiskit_result_options' : result_options,
        }

        tomography_results = self._run_pure_kraus_series_tomography(
                pure_state_kraus_circuits,
                t_params,
                backend,
                shots,
                circuit_weights=t_weights,
                renormalize=renormalize,
                positive_definite=positive_definite,
                **primitive_kwargs)

        if 'submit_jobs_only' in primitive_kwargs and primitive_kwargs['submit_jobs_only']:
            return tomography_results

        return self.unitary_transform.conj().T @ \
                    tomography_results @ \
                    self.unitary_transform


    def simulate_classical_density_matrix_evolution(self,
                                                    initial_state,
                                                    t,
                                                    observable=None):
        """Simulates the density matrix trajectory of the system (or an observable)
           by classically solving the Lindblad equation.

        Args:
            initial_state (np.ndarray): Initial state of the system (if 1D, assumes a statevector;
                                        or if 2D, assumes a density matrix). The dimensions of the
                                        initial state must be of size 2^(n_qubits).
            t (np.ndarray): Trajectory time values to sample during evolution.
            observable (np.ndarray, optional): If not None, the expectation value of this observable is returned
                                               for each t value in the trajectory. Defaults to None.

        Returns:
            np.ndarray: List of density matrices (or observable expectation values)
                        for each time t in the trajectory.
        """
        
        gamma_xyz_norm = np.linalg.norm(self.gamma_xyz)
        gamma_n = self.gamma_xyz / gamma_xyz_norm if gamma_xyz_norm > 0 else self.gamma_xyz

        if self.simultaneous:
            J_n = gamma_n[0]*self.J_x_operator() + \
                gamma_n[1]*self.J_y_operator() + \
                gamma_n[2]*self.J_z_operator()
            lindblad_ops = [ J_n ]
            gammas = [ gamma_xyz_norm ]
        else:
            lindblad_ops = [
                self.J_x_operator(),
                self.J_y_operator(),
                self.J_z_operator()
            ]
            gammas = self.gamma_xyz

        rhos = simulate_lindblad_trajectory(initial_state, t,
                                           hamiltonian=self.hamiltonian_operator(),
                                           lindblad_ops=lindblad_ops,
                                           gammas=gammas,
                                           hbar=self.hbar)
        
        # compute expectation value of observable (if given):
        if observable is not None:
            return np.real(np.trace(observable @ rhos, axis1=1,axis2=2))
        
        return rhos

    def _plot_density_coutours(self, X, Y, Z, xlabel, ylabel, ax=None, levels=100, cmap='plasma'):
        new_plot = (ax is None)
        if new_plot:
            plt.figure()
            ax = plt.gca()
        
        cm = ax.contourf(X,Y,Z, levels=levels, cmap=cmap, vmin=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if new_plot:
            plt.show()

    def get_position_density(self, density_matrix, m=1.0, xlim=(-2,2), ylim=(-2,2), mesh_size=(100,100), unitless=False):
        rho_probs, rho_states = np.linalg.eigh(density_matrix)
        rho_probs = np.array(rho_probs)
        rho_states = np.array(rho_states)

        x0_scale = np.sqrt(self.hbar / (m*self.omega))
        x = np.linspace(*xlim, mesh_size[0])
        y = np.linspace(*ylim, mesh_size[1])

        if unitless:
            x = x.copy()*x0_scale
            y = y.copy()*x0_scale

        X, Y = np.meshgrid(x, y)
        qho_states = range(2**self.n_qubits)

        xy_amplitudes = np.array([
            qho_eigenstate(nx,X, omega=self.omega, m=m, hbar=self.hbar) * \
            qho_eigenstate(ny,Y, omega=self.omega, m=m, hbar=self.hbar)
            for nx in qho_states
            for ny in qho_states
        ])

        U_amplitudes = np.einsum('jk,jlm->klm', rho_states, xy_amplitudes)
        densities = np.einsum('k,klm->lm', rho_probs, np.abs(U_amplitudes)**2)

        if unitless:
            X /= x0_scale
            Y /= x0_scale

        return X, Y, densities


    def get_momentum_density(self, density_matrix, m=1.0, pxlim=(-2,2), pylim=(-2,2), mesh_size=(100,100), unitless=False):
        rho_probs, rho_states = np.linalg.eigh(density_matrix)
        rho_probs = np.array(rho_probs)
        rho_states = np.array(rho_states)
        
        p0_scale = np.sqrt(self.hbar*m*self.omega)
        px = np.linspace(*pxlim, mesh_size[0])
        py = np.linspace(*pylim, mesh_size[1])

        if unitless:
            px = px.copy()*p0_scale
            py = py.copy()*p0_scale

        Px, Py = np.meshgrid(px, py)
        qho_states = range(2**self.n_qubits)

        p_xy_amplitudes = np.array([
            (-1.j)**nx * qho_eigenstate(nx,Px, omega=1/(m*m*self.omega), m=m, hbar=self.hbar) * \
            (-1.j)**ny * qho_eigenstate(ny,Py, omega=1/(m*m*self.omega), m=m, hbar=self.hbar)
            for nx in qho_states
            for ny in qho_states
        ])

        if unitless:
            Px /= p0_scale
            Py /= p0_scale

        U_amplitudes = np.einsum('jk,jlm->klm', rho_states, p_xy_amplitudes)
        densities = np.einsum('k,klm->lm', rho_probs, np.abs(U_amplitudes)**2)

        return Px, Py, densities

    def plot_position_density(self, density_matrix, m=1.0, xlim=(-2,2), ylim=(-2,2), mesh_size=(100,100), ax=None, cmap='plasma', unitless=False):
        
        
        X, Y, Z = self.get_position_density(
            density_matrix,
            m=m,
            xlim=xlim,
            ylim=ylim,
            mesh_size=mesh_size,
            unitless=unitless
        )
        self._plot_density_coutours(X=X,Y=Y,Z=Z, 
                                    xlabel=r'$x_0$' if unitless else r'$x$',
                                    ylabel=r'$y_0$' if unitless else r'$y$',
                                    ax=ax,
                                    levels=100,
                                    cmap=cmap)

    def plot_position_density_animation(self, t, density_matrices, m=1.0, xlim=(-2,2), ylim=(-2,2), mesh_size=(100,100),
                                        overlay=True, fig=None, ax=None, cmap='plasma', levels=10, unitless=False):

        new_plot = (ax is None or fig is None)
        if new_plot:
            fig, ax = plt.subplots()

        density_kwargs = dict(
            m=m,
            xlim=xlim,
            ylim=ylim,
            mesh_size=mesh_size,
            unitless=unitless
        )
        
        t = np.array(t)
        X, Y, Z_0 = self.get_position_density(density_matrices[0],**density_kwargs)
        Z_t = np.array([
            self.get_position_density(rho, **density_kwargs)[2]
            for rho in density_matrices
        ])

        cax = ax.pcolormesh(X,Y,Z_0, cmap=cmap, vmin=0, shading='gouraud')
        timelabel = ax.text(0.9,0.9, "", transform=ax.transAxes, ha="right", 
                            bbox=dict(boxstyle='round', facecolor='white'))
        ax.set_xlabel(r'$x_0$' if unitless else r'$x$')
        ax.set_ylabel(r'$y_0$' if unitless else r'$y$')

        def _animate_position_density(frame):
            cax.set_array(Z_t[frame])
            timelabel.set_text(f't = {t[frame]:.3f}')

            return cax, timelabel


        anim = animation.FuncAnimation(fig, _animate_position_density, frames=len(t), interval=5)

        return anim

        
    def plot_momentum_density(self, density_matrix, m=1.0, pxlim=(-2,2), pylim=(-2,2), mesh_size=(100,100), ax=None, cmap='plasma', unitless=False):
        
        
        Px, Py, Z = self.get_momentum_density(
            density_matrix,
            m=m,
            pxlim=pxlim,
            pylim=pylim,
            mesh_size=mesh_size,
            unitless=unitless
        )
        self._plot_density_coutours(X=Px,Y=Py,Z=Z, 
                                    xlabel=r'$p_{x0}$' if unitless else r'$p_x$', 
                                    ylabel=r'$p_{y0}$' if unitless else r'$p_y$', 
                                    ax=ax,
                                    levels=100,
                                    cmap=cmap)

    def plot_momentum_density_animation(self, t, density_matrices, m=1.0, pxlim=(-2,2), pylim=(-2,2), mesh_size=(100,100), 
                                        fig=None, ax=None, cmap='plasma', unitless=False):

        new_plot = (ax is None or fig is None)
        if new_plot:
            fig, ax = plt.subplots()

        density_kwargs = dict(
            m=m,
            pxlim=pxlim,
            pylim=pylim,
            mesh_size=mesh_size,
            unitless=unitless
        )
        
        t = np.array(t)
        X, Y, Z_0 = self.get_momentum_density(density_matrices[0],**density_kwargs)
        Z_t = np.array([
            self.get_momentum_density(rho, **density_kwargs)[2]
            for rho in density_matrices
        ])

        cax = ax.pcolormesh(X,Y,Z_0, cmap=cmap, vmin=0, shading='gouraud')
        timelabel = ax.text(0.9,0.9, "", transform=ax.transAxes, ha="right",
                            bbox=dict(boxstyle='round', facecolor='white'))
        ax.set_xlabel(r'$p_{x0}$' if unitless else r'$p_x$')
        ax.set_ylabel(r'$p_{y0}$' if unitless else r'$p_y$')

        def _animate_position_density(frame):
            cax.set_array(Z_t[frame])
            timelabel.set_text(f't = {t[frame]:.3f}')

            return cax, timelabel


        anim = animation.FuncAnimation(fig, _animate_position_density, frames=len(t), interval=5)

        return anim


            
    

        
        


        