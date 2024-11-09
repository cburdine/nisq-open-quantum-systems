import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import XGate, RYGate, GlobalPhaseGate, PhaseGate
from qiskit.circuit.library import StatePreparation

from open_quantum_systems.math_util import *
from open_quantum_systems.lindblad import simulate_lindblad_trajectory
from open_quantum_systems.tomography import KrausSeriesEstimator, KrausSeriesSampler, KrausSeriesTomography

from scipy.linalg import expm
from scipy.special import factorial, hermite, genlaguerre

def qho_eigenstate(n, x, omega=1., m=1., hbar=1.):
    """Evaluates the wavefunction of the nth eigenstate of a 1D quantum harmonic oscilator
       at a position x in space.

    Args:
        n (int): Eigenstate index.
        x (float, np.ndarray): Position in space (can be a numpy array).
        omega (float, optional): Angular frequency of oscillator. Defaults to 1..
        m (float, optional): Mass of oscillator. Defaults to 1..
        hbar (float, optional): Reduced planck constant. Defaults to 1..

    Returns:
        float, np.ndarray: The amplitude of the wave function at position(s) in space.
    """
    alpha = np.sqrt(m*omega/hbar)
    A = np.sqrt(alpha / (2**n * factorial(n) * np.sqrt(np.pi)))
    return A*np.exp(-(alpha*x)**2 / 2) * hermite(n)(alpha*x)

def qho_ladder(n_qubits=2):
    """Generates a Quantum Harmonic ladder operator "a" in matrix form 

    Args:
        n_qubits (int, optional): Number of qubits. Defaults to 2.

    Returns:
        np.ndarray: Ladder operator in matrix form
    """
    a = np.zeros((2**n_qubits,2**n_qubits))
    for i in range(1,2**n_qubits):
        a[i-1,i] = np.sqrt(i)

    return a

def qho_coherent_state(alpha=1., n_qubits=2):
    """Generates a coherent harmonic oscillator statevector 
        (see https://en.wikipedia.org/wiki/Coherent_state)

    Args:
        alpha (complex, optional): Alpha coherent state parameter. Defaults to 1..
        n_qubits (int, optional): Number of qubits. Defaults to 2.

    Returns:
        np.ndarray: Statevector of coherent state
    """
    a = qho_ladder(n_qubits)
    alpha_psi = np.exp(-alpha*np.conj(alpha)/2)* (expm(alpha * a.conj().T) @ expm(-np.conj(alpha)*a))
    psi = alpha_psi[:,0]
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    return psi.astype(np.complex128)

def qho_cat_state(alpha=1., n_qubits=2, theta=np.pi):
    """Generates a cat state harmonic oscillator statevector 
            (https://en.wikipedia.org/wiki/Cat_state)

        This state takes the form:

        |a> + exp(i * theta)|a>

        where a (alpha) is a coherent state and theta is the phase angle.

    Args:
        alpha (complex, optional): Alpha coherent state parameter. Defaults to 1..
        n_qubits (int, optional): Number of qubits. Defaults to 2.

    Returns:
        np.ndarray: Statevector of cat state
    """
    ket_a_plus = qho_coherent_state(alpha=alpha, n_qubits=n_qubits)
    ket_a_minus = qho_coherent_state(alpha=-alpha, n_qubits=n_qubits)

    psi = ket_a_plus + np.exp(1.j * theta)*ket_a_minus
    psi /= np.sqrt(np.sum(np.abs(psi)**2))

    return psi

def make_increment_gate(n_qubits, control=False):
    """Makes a Multi-qubit increment circuit gate that maps |x> to |x+1>

    Args:
        n_qubits (int): Number of qubits
        control (bool): If True, adds an additional control qubit (qubit 0). Defaults to False.

    Returns:
        qiskit.circuit.instruction.Instruction: Quantum Circuit Gate
    """
    if control:
        qc = QuantumCircuit(n_qubits+1, name='ctrl_inc')
        ctrl_qubit = 0
        for i in range(n_qubits,0,-1):
            carry_ctrl = list(range(i))
            carry_gate = XGate().control(len(carry_ctrl)) if carry_ctrl else XGate()
            qc.append(carry_gate, carry_ctrl + [i])
        return qc.to_instruction()
    
    else:
        qc = QuantumCircuit(n_qubits, name='inc')
        for i in range(n_qubits-1,-1,-1):
            carry_ctrl = list(range(i))
            carry_gate = XGate().control(len(carry_ctrl)) if carry_ctrl else XGate()
            qc.append(carry_gate, carry_ctrl + [i])

        return qc.to_instruction()

def make_decrement_gate(n_qubits, control=False):
    """Makes a Multi-qubit decrement circuit gate that maps |x> to |x-1>

    Args:
        n_qubits (int): Number of qubits
        control (bool): If True, adds an additional control qubit (qubit 0). Defaults to False.

    Returns:
        qiskit.circuit.instruction.Instruction: Quantum Circuit Gate
    """
    qc = make_increment_gate(n_qubits, control).inverse()
    qc.name = 'ctrl_dec' if control else 'dec'
    return qc

def make_add_n_gate(n_qubits, n):
    """Makes a multi-qubit subtraction gate that maps |x> to |x-n>

    Args:
        n_qubits (int): Number pf qubits
        n (int): The number to be subtracted

    Returns:
        qiskit.circuit.instruction.Instruction: Quantum Circuit Gate
    """

    qc = QuantumCircuit(n_qubits, name=f'Add({n})')
    for i in range(n_qubits):
        if (n & (1<<i)):
            inc_i = make_increment_gate(n_qubits-i)
            qc.append(inc_i, list(range(i,n_qubits)))
    return qc.decompose().to_instruction()

def make_sub_n_gate(n_qubits, n):
    """Makes a multi-qubit addition gate that maps |x> to |x+n>

    Args:
        n_qubits (int): Number pf qubits
        n (int): The number to be added

    Returns:
        qiskit.circuit.instruction.Instruction: Quantum Circuit Gate
    """
    qc = QuantumCircuit(n_qubits, name=f'Sub({n})')
    for i in range(n_qubits):
        if (n & (1<<i)):
            dec_i = make_decrement_gate(n_qubits-i)
            qc.append(dec_i, list(range(i,n_qubits)))
    return qc.decompose().to_instruction()
            

    
def make_theta_transform(n_qubits):
    """Makes a matrix mapping parameterized controlled rotations in a circuit
    to theta values (i.e arctangents of the diagonal of an operator).

    Args:
        n_qubits (int): Number of qubits

    Returns:
        np.ndarray: Transform matrix
    """
    F1 = np.array([[1,0],
                   [1,1]])
    Fn = np.ones((1,1))
    for i in range(n_qubits):
        Fn = np.kron(Fn,F1)

    return Fn

def make_inv_theta_transform(n_qubits):
    """Makes a matrix mapping theta values (i.e arctangents of the 
    diagonal of an operator) to parameterized controlled rotations.
    This is the inverse of the matrix generated by make_theta_transform

    Args:
        n_qubits (int): Number of qubits

    Returns:
        np.ndarray: Transform matrix
    """
    
    F1_inv = np.array([[1,0],
                       [-1,1]])
    Fn_inv = np.ones((1,1))
    for i in range(n_qubits):
        Fn_inv = np.kron(Fn_inv,F1_inv)

    return Fn_inv

def _make_controlled_ry(theta, n):
    """ This circumvents a bug in Qiskit's synthesis of controlled parameterized gates with n > 3"""
    
    if n <= 3:
        return RYGate(theta).control(n)
    else:
        ctrl_qubits = list(range(n))
        ry_3 = RYGate(theta/2).control(3)
        ccnx = XGate().control(n-3)
        
        qc = QuantumCircuit(n+1, name=f'Ry_ctrl{n}')
        qc.append(ry_3, ctrl_qubits[-3:]+[n])
        qc.append(ccnx, ctrl_qubits[:-3]+[n])
        qc.append(ry_3.inverse(), ctrl_qubits[-3:]+[n])
        qc.append(ccnx, ctrl_qubits[:-3]+[n])

        return qc.to_instruction()

def make_diagonal_contraction_circuit(n_qubits, diag=None, thetas=None, theta_tolerance=1e-6):
    """Makes a QuantumCircuit implementing a minimal unitary dilation of a diagonal contraction operator.

    Args:
        n_qubits (int): number of qubits in the diagonal operation
        diag (np.ndarray, optional): Diagonal of the contraction operator. Defaults to None.
        thetas (qiskit.circuit.ParameterVector, optional): ParameterVector of controlled rotation parameters. Defaults to None.
        theta_tolerance (np.ndarray, optional): Tolerance for parameterized rotations. If increased, generates fewer gates but lowers accuracy. Defaults to 1e-6.

    Returns:
        qiskit.circuit.QuantumCircuit: Diagonal contraction operator quantum circuit
    """

    if diag is None:
        # create and pad thetas to correct size:
        if thetas is None:
            thetas =  ParameterVector('theta',2**n_qubits) #[ Parameter(f'theta_{i}') for i in range(2**n_qubits - order) ] 
        if len(thetas) < 2**n_qubits:
            assert(isinstance(thetas, list))
            thetas += [0]*(2**n_qubits - len(thetas))
    else:
        diag = np.array(diag)

        # compute controlled rotation angles:
        betas = np.zeros(2**n_qubits, dtype=diag.dtype)
        betas[:diag.shape[0]] = diag
        thetas = make_inv_theta_transform(n_qubits) @ (-2*np.arcsin(betas))
        thetas[np.abs(thetas) < np.abs(theta_tolerance)] = 0.0

    qc = QuantumCircuit(n_qubits+1, name=r'$L_n$ Diagonal')

    qc.x(n_qubits)
    for i, theta in enumerate(thetas):
        if theta != 0:
            i_bits = "{0:b}".format(i)
            ctrl_bits = [ j for j, bit in enumerate(reversed(i_bits)) if bit == '1' ]
            ctrl_RY = RYGate(theta)
            if ctrl_bits:
                ctrl_RY = _make_controlled_ry(theta, len(ctrl_bits))
                
            qc.append(ctrl_RY, ctrl_bits+[n_qubits])
        
    return qc


def make_ladder_dilation_circuit(n_qubits, order=1, superdiag=None, thetas=None, theta_tolerance=1e-6):
    """Makes a QuantumCircuit implementing a minimal unitary dilation of a
       ladder operator raised to a power (the order of the ladder operator).

    Args:
        n_qubits (int): Number of qubits
        order (int, optional): Order of the operator. Defaults to 1.
        superdiag (np.ndarray, optional): Superdiagonal of the operator. Defaults to None.
        thetas (qiskit.circuit.ParameterVector, optional): ParameterVector of controlled rotation parameters. Defaults to None.
        theta_tolerance (np.ndarray, optional): Tolerance for parameterized rotations. If increased, generates fewer gates but lowers accuracy. Defaults to 1e-6.

    Returns:
        qiskit.circuit.QuantumCircuit: Ladder operator quantum circuit
    """
    
    if superdiag is None:
        # create and pad thetas to correct size:
        if thetas is None:
            thetas =  ParameterVector('theta',2**n_qubits) #[ Parameter(f'theta_{i}') for i in range(2**n_qubits - order) ] 
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
    
    qc = QuantumCircuit(n_qubits+1, name=r'$L_n$ Decoherence')

    # add subtraction gate:
    if order > 0:
        sub_n = make_sub_n_gate(n_qubits, order)
        qc.append(sub_n, list(range(n_qubits)))


    qc.x(n_qubits)
    for i, theta in enumerate(thetas):
        if theta != 0:
            i_bits = "{0:b}".format(i)
            ctrl_bits = [ j for j, bit in enumerate(reversed(i_bits)) if bit == '1' ]
            ctrl_RY = RYGate(theta)
            if ctrl_bits:
                ctrl_RY = _make_controlled_ry(theta, len(ctrl_bits)) # ctrl_RY.control(len(ctrl_bits))
                
            qc.append(ctrl_RY, ctrl_bits+[n_qubits])
        
    return qc
    
def standard_angles(theta):
    """Maps angles (in radians) to [-pi,pi]

    Args:
        theta (np.ndarray): Angles in radians

    Returns:
        np.ndarray: Standardized angles
    """
    return np.fmod(theta + np.pi, 2*np.pi) - np.pi

def make_diagonal_hamiltonian_circuit(n_qubits, H_diag=None, t=1.0, phis=None, conj=False, phi_tolerance=1e-10):
    """Makes a QuantumCircuit implementing the time-evolution of a diagonal Hamiltonian

    Args:
        n_qubits (int): Number of qubits
        H_diag (np.ndarray, optional): Diagonal of Hamiltonian (uses standard QHO Hamiltonian if None). Defaults to None.
        t (float, optional): _description_. Defaults to 1.0.
        phis (list[qiskit.circuit.ParameterVector], optional): ParameterVector of phase parameters. Defaults to None.
        conj (bool, optional): If True returns the conjugate Hamiltonian circuit (i.e. negative time evolution). Defaults to False.
        phi_tolerance (float, optional): Tolerance for phase parameters. If increased, generates fewer gates but lowers accuracy. Defaults to 1e-10.

    Returns:
        qiskit.circuit.QuantumCircuit: Hamiltonian time-evolution circuit
    """

    qc = QuantumCircuit(n_qubits, name=r'$\exp(-it H /\hbar)$')
    if H_diag is None:
        # create phis:
        if phis is None:
            phis =  ParameterVector('phi', 2**n_qubits)
    else:
        H_diag = np.array(H_diag)

        # compute controlled rotation angles:
        phis = make_inv_theta_transform(n_qubits) @ standard_angles(t*H_diag)
        phis[np.abs(phis) < np.abs(phi_tolerance)] = 0

    for i, phi in enumerate(phis):
        if phi != 0:
            if i == 0:
                qc.append(GlobalPhaseGate(-phi) if conj else GlobalPhaseGate(phi))
            else:
                i_bits = "{0:b}".format(i)
                ctrl_bits = [ j for j, bit in enumerate(reversed(i_bits)) if bit == '1' ]
                ctrl_P = PhaseGate(-phi) if conj else PhaseGate(phi)
                if len(ctrl_bits) > 1:
                    ctrl_P = ctrl_P.control(len(ctrl_bits)-1)
                qc.append(ctrl_P, ctrl_bits)
    
    return qc

def make_qho_kraus_circuit_masks(n_qubits, kraus_order=2):
    """Makes a list of operator masks that project operators into the
       the minimal subspace where measurements must be taken to compute
       operator expectation values


    Args:
        n_qubits (int): Number of qubits
        kraus_order (int, optional): Order of the kraus series used. Defaults to 2.

    Returns:
        list: List of Kraus operator masks up to the given order.
    """
    masks = []
    for order in range(kraus_order+1):
        order_mask = np.ones(2**n_qubits)
        if order > 0:
            order_mask[-order:] = 0
        
        masks.append(order_mask)

    return masks
        

def make_parameterized_qho_circuits(n_qubits,
                                    state_prep_circuit=None,
                                    kraus_order=2):
    """Makes a list of parameterized QuantumCircuits for simulating a damped quantum harmonic oscillator.
       These circuits evolve an initial pure state into state that is entangled with an ancillary environmental qubit.

    Args:
        n_qubits (int): Number of qubits
        state_prep_circuit (qiskit.circuit.QuantumCircuit, optional): Pure state preparation circuit. Defaults to None.
        kraus_order (int, optional): Order of the Kraus series used to simulate the system. Defaults to 2.

    Returns:
        tuple: Tuple of: (circuits, conjugate circuits, theta Parameters, phi parameters)
    """

    circuits = []
    conj_circuits = []
    theta_params = ParameterVector('theta',2**n_qubits)
    phi_params = ParameterVector('phi',2**n_qubits)
    
    for order in range(kraus_order+1):
        
        # create circuit:
        n_prep_qubits = 0 if state_prep_circuit is None else len(state_prep_circuit.qubits)
        order_qc = QuantumCircuit(max(n_qubits+1,n_prep_qubits+1), name=f'qho_k[{order}]')
        conj_order_qc = QuantumCircuit(max(n_qubits+1,n_prep_qubits+1), name=f'qho_k[{order}]_conj')
        
        # append state preparation circuit:
        if state_prep_circuit is not None:
            prep_ancillas = [i+1 for i in range(n_qubits, n_prep_qubits)]
            prep_qubits = list(range(n_qubits)) + prep_ancillas
            order_qc.append(state_prep_circuit, prep_qubits)

        # apply dissipative operator (this is real, so it is its own conjugate):
        ladder_circuit = make_ladder_dilation_circuit(
            n_qubits=n_qubits,
            order=order,
            thetas=theta_params
        )
        order_qc.append(ladder_circuit, list(range(n_qubits+1)))
        conj_order_qc.append(ladder_circuit, list(range(n_qubits+1)))

        # evolve dissipated state via Hamiltonian simulation:
        hamiltonian_circuit = make_diagonal_hamiltonian_circuit(
            n_qubits=n_qubits,
            phis=phi_params)
        conj_hamiltonian_circuit = make_diagonal_hamiltonian_circuit(
            n_qubits=n_qubits,
            phis=phi_params,
            conj=True)
        order_qc.append(hamiltonian_circuit,list(range(n_qubits)))
        conj_order_qc.append(conj_hamiltonian_circuit, list(range(n_qubits)))
        
        circuits.append(order_qc)
        conj_circuits.append(conj_order_qc)

    return circuits, conj_circuits, theta_params, phi_params

def get_qho_timeseries_parameters(n_qubits,
                                t_vals,
                                theta_params,
                                phi_params,
                                omega=1.0,
                                gamma=1.0,
                                hbar=1.0,
                                kraus_order=2,
                                angle_tolerance=1e-10):
    """Generates the parameters for time-series evolution of a quantum harmonic oscillator.

    Args:
        n_qubits (int): Number of qubits
        t_vals (np.ndarray): Time values in trajectory
        theta_params (qiskit.circuit.ParameterVector): ParameterVector of theta parameters (for ladder dilation circuit)
        phi_params (qiskit.circuit.ParameterVector): ParameterVector of phi parameters (for Hamiltonian time evolution)
        omega (float, optional): Angular frequency of oscillator. Defaults to 1.0.
        gamma (float, optional): Coupling constant of damping Lindblad operator. Defaults to 1.0.
        hbar (float, optional): Value of the reduced Planck constant. Defaults to 1.0.
        kraus_order (int, optional): Order of the Kraus operator series used. Defaults to 2.
        angle_tolerance (float, optional): Tolerance in radians allowed for circuit rotations (higher tolerance may give smaller circuits but less accuracy). Defaults to 1e-10.
        
    Returns:
        list[dict]: List of Parameter dictionary for each time t in t_vals
    """

    N = np.arange(2**n_qubits)
    H_diag = hbar*omega*(0.5 + N)
    parameters = []

    for t in t_vals:

        # compute phi (Hamiltonian evolution phase parameters):
        phis = make_inv_theta_transform(n_qubits) @ standard_angles(-t*H_diag/hbar)
        phis[np.abs(phis) < np.abs(angle_tolerance)] = 0
        phi_values = {phi_params[i]: phis[i] for i in range(2**n_qubits)}

        # compute theta parameters (these encode the superdiagonal of the dilated ladder operators):
        parameter_values = []
        for order in range(kraus_order+1):
            
            betas = np.exp(-0.5*gamma*t*N)
            betas *= np.sqrt((gamma*t + gamma*gamma*zassenhaus_g(t,v=gamma))**order / factorial(order))
            for i in range(1,order+1):
                betas[:-i] *= np.sqrt(N[i:])
                betas[-i:] = 0

            if np.max(np.abs(betas)) > 1:
                raise ValueError(f'Order {order} Kraus operator has norm exceeding one; check t values and operator superdiagonal for invalid (i.e. negative or complex) values')
            
            thetas = make_inv_theta_transform(n_qubits) @ (-2*np.arcsin(betas))
            thetas[np.abs(thetas) < np.abs(angle_tolerance)] = 0.0
            theta_values = {theta_params[i] : thetas[i] for i in range(2**n_qubits)}
        
            parameter_values.append(phi_values | theta_values)

        parameters.append(parameter_values)
        
    return parameters

class QHOSimulation:
    """ Object for a Kraus Series-based simulation of a damped quantum harmonic oscillator"""    
    def __init__(self, 
                 n_qubits=2, 
                 kraus_order=2,
                 hbar=1.0,
                 omega=1.0,
                 gamma=1.0,
                 angle_tolerance=1e-10):
        """Constructs a quantum harmonic oscillator simulation

        Args:
            n_qubits (int, optional): Number of qubits used to represent the system. Defaults to 2.
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
        self.gamma = gamma
        self.angle_tolerance = angle_tolerance

        self.kraus_circuit_masks = \
            make_qho_kraus_circuit_masks(n_qubits, kraus_order)
        
        self.kraus_circuits, \
            self.conj_kraus_circuits, \
            self.theta_params, \
            self.phi_params = \
            make_parameterized_qho_circuits(n_qubits=n_qubits,
                                            state_prep_circuit=None,
                                            kraus_order=kraus_order)

    def energies(self):
        """Returns the energies of eigenstates of the QHO simulation

        Returns:
            np.ndarray: Array of energy values
        """
        H_diag = self.hbar*self.omega*(0.5 + np.arange(2**self.n_qubits))
        return H_diag

    def hamiltonian_operator(self):
        """Returns the Hamiltonian operator of the QHO simulation

        Returns:
            np.ndarray: Square matrix operator representing the Hamiltonian
        """
        return np.diag(self.energies())

    def ladder_operator(self):
        """Returns the annihilation operator "a" of the QHO simulation.

        Returns:
            np.ndarray: Square matrix operator representing the annihilation operator
        """
        return qho_ladder(self.n_qubits)

    def position_operator(self, m=1.0, unitless=False):
        """Returns the position operator of the QHO simulation for an oscillating mass m.

        Args:
            m (float, optional): The oscillating mass. Defaults to 1.0.

        Returns:
            np.ndarray: Square matrix operator representing the position operator
        """
        a = qho_ladder(self.n_qubits)
        x0_scale = np.sqrt(self.hbar / (m*self.omega))
        x = np.sqrt((self.hbar)/(2*m*self.omega))*(a.conj().T + a)
        return x / x0_scale if unitless else x

    def momentum_operator(self, m=1.0, unitless=False):
        """Returns the momentum operator of the QHO simulation for an oscillating mass m.

        Args:
            m (float, optional): The oscillating mass. Defaults to 1.0.

        Returns:
            np.ndarray: Square matrix operator representing the momentum operator
        """
        a = qho_ladder(self.n_qubits)
        p0_scale = np.sqrt(self.hbar*m*self.omega)
        p = 1.j*np.sqrt((self.hbar*m*self.omega)/2)*(a.conj().T - a)
        return p / p0_scale if unitless else p

    def _run_pure_kraus_circuit_sampler(self,
                                        circuits,
                                        t_params,
                                        backend,
                                        shots,
                                        masking=True,
                                        renormalize=True,
                                        submit_jobs_only=False,
                                        submitted_jobs_data=None,
                                        **sampler_kwargs):
        
        sampler = KrausSeriesSampler(backend, **sampler_kwargs)
        result_masks = self.kraus_circuit_masks if masking else None

        # launch jobs:
        if submitted_jobs_data is None:
            submitted_jobs_data = sampler.launch_jobs(
                circuits,
                observable_qubits=list(range(self.n_qubits)),
                conditional_qubits=[self.n_qubits],
                shots_per_circuit=shots,
                parameters=t_params,
                result_masks=result_masks,
                circuit_weights=None,
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
                            conditional_qubits=[self.n_qubits],
                            shots_per_circuit=shots,
                            parameters=t_params,
                            result_masks=result_masks,
                            circuit_weights=None,
                            renormalize=renormalize)

        sampler_probs = np.zeros((len(sampler_results),2**self.n_qubits))
        for i, prob_dict in enumerate(sampler_results):
            for j, p in prob_dict.items():
                sampler_probs[i,j] = p
        
        return sampler_probs

    def _run_pure_kraus_circuit_estimator(self,
                                          circuits,
                                          t_params,
                                          backend,
                                          observable,
                                          shots,
                                          masking=True,
                                          renormalize=True,
                                          submit_jobs_only=False,
                                          submitted_jobs_data=None,
                                          **estimator_kwargs
                                          ):
        estimator = KrausSeriesEstimator(backend, **estimator_kwargs)
        result_masks = self.kraus_circuit_masks if masking else None
        
        # launch jobs:
        if submitted_jobs_data is None:
            submitted_jobs_data = estimator.launch_jobs(
                circuits,
                observable,
                observable_qubits=list(range(self.n_qubits)),
                conditional_qubits=[self.n_qubits],
                shots_per_circuit=shots,
                parameters=t_params,
                result_masks=result_masks,
                circuit_weights=None,
                renormalize=renormalize
            )
        
        # return job data if requested:
        if submit_jobs_only:
            return submitted_jobs_data

        # process job results:
        job_array, job_pauli_array = submitted_jobs_data
        eval_results = np.real_if_close(estimator.get_results(
            job_array,
            job_pauli_array,
            circuits,
            observable,
            observable_qubits=list(range(self.n_qubits)),
            conditional_qubits=[self.n_qubits],
            shots_per_circuit=shots,
            parameters=t_params,
            result_masks=result_masks,
            circuit_weights=None,
            renormalize=renormalize
        ))

        return eval_results

    def _run_pure_kraus_series_tomography(self,
                                          circuits,
                                          t_params,
                                          backend,
                                          shots,
                                          masking=True,
                                          renormalize=True,
                                          positive_definite=True,
                                          submit_jobs_only=False,
                                          submitted_jobs_data=None,
                                          **tomography_kwargs):
        
        tomography = KrausSeriesTomography(backend, **tomography_kwargs)
        result_masks = self.kraus_circuit_masks if masking else None

        # launch jobs:
        if submitted_jobs_data is None:
            submitted_jobs_data = tomography.launch_jobs(
                circuits,
                observable_qubits=list(range(self.n_qubits)),
                conditional_qubits=[self.n_qubits],
                shots_per_circuit=shots,
                parameters=t_params,
                result_masks=result_masks,
                circuit_weights=None,
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
            conditional_qubits=[self.n_qubits],
            shots_per_circuit=shots,
            parameters=t_params,
            result_masks=result_masks,
            circuit_weights=None,
            renormalize=renormalize,
            positive_definite=positive_definite)

        return tomography_results

    def simulate_pure_state_evolution(self,
                                      pure_state,
                                      t,
                                      backend,
                                      observable=None,
                                      shots=1024,
                                      masking=True,
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
            masking (bool, optional): If True, suppresses "impossible" measurement outcomes for
                                      each Kraus operator. Defaults to True.
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
        t_params = get_qho_timeseries_parameters(
            n_qubits=self.n_qubits,
            t_vals=t,
            gamma=self.gamma,
            theta_params=self.theta_params,
            phi_params=self.phi_params,
            omega=self.omega,
            hbar=self.hbar,
            kraus_order=self.kraus_order,
            angle_tolerance=self.angle_tolerance)

        # convert pure state to numpy array:
        pure_state = np.array(pure_state).squeeze()
        
        # prepend state preparation circuit:
        pure_state_kraus_circuits = []
        for kraus_circuit in self.kraus_circuits:
            pure_qc = QuantumCircuit(len(kraus_circuit.qubits), name=kraus_circuit.name)
            state_prep_gate = StatePreparation(pure_state)
            pure_qc.append(state_prep_gate, list(range(self.n_qubits)))
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
                masking=masking,
                renormalize=renormalize,
                **primitive_kwargs)
        else:
            return self._run_pure_kraus_circuit_estimator(
                pure_state_kraus_circuits,
                t_params,
                backend,
                observable,
                shots,
                masking=masking,
                renormalize=renormalize,
                **primitive_kwargs)

    def simulate_pure_state_density_evolution(self, 
        pure_state,
        t,
        backend,
        shots=1024,
        masking=True,
        renormalize=True,
        positive_definite=True,
        transpile_options={},
        run_options={},
        result_options={},
        primitive_kwargs={}):
        """Simulates the trajectory evolution of the system initialized in a pure state.

        Args:
            pure_state (np.ndarray): Initial pure state of the system (in the energy basis). Must be of length 2^(n_qubits).
            t (np.ndarray): Trajectory time values to sample during evolution.
            backend (qiskit.providers.Backend): Backend to use for the simulation.
            shots (int, optional): Number of shots to use per quantum circuit. Defaults to 1024.
            masking (bool, optional): If True, suppresses "impossible" measurement outcomes for 
                                      each Kraus operator. Defaults to True.
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
        t_params = get_qho_timeseries_parameters(
            n_qubits=self.n_qubits,
            t_vals=t,
            gamma=self.gamma,
            theta_params=self.theta_params,
            phi_params=self.phi_params,
            omega=self.omega,
            hbar=self.hbar,
            kraus_order=self.kraus_order,
            angle_tolerance=self.angle_tolerance)

        # convert pure state to numpy array:
        pure_state = np.array(pure_state).squeeze()
        
        # prepend state preparation circuit:
        pure_state_kraus_circuits = []
        for kraus_circuit in self.kraus_circuits:
            pure_qc = QuantumCircuit(len(kraus_circuit.qubits), name=kraus_circuit.name)
            state_prep_gate = StatePreparation(pure_state)
            pure_qc.append(state_prep_gate, list(range(self.n_qubits)))
            pure_qc.append(kraus_circuit, list(range(len(kraus_circuit.qubits))))
            pure_state_kraus_circuits.append(pure_qc.decompose())
        

        primitive_kwargs |= {
            'qiskit_transpile_options' : transpile_options,
            'qiskit_run_options' : run_options,
            'qiskit_result_options' : result_options,
        }

        return self._run_pure_kraus_series_tomography(
            pure_state_kraus_circuits,
                t_params,
                backend,
                shots,
                masking=masking,
                renormalize=renormalize,
                positive_definite=positive_definite,
                **primitive_kwargs)
    
    def simulate_full_density_evolution(self,
        initial_state,
        t,
        backend,
        shots=1024,
        magnitudes_only=True,
        masking=True,
        renormalize=True,
        transpile_options={},
        run_options={},
        result_options={},
        primitive_kwargs={}):

        raise NotImplementedError('This function is not yet implemented. :(')

        # calculate circuit parameters for each timestep:
        t_params = get_qho_timeseries_parameters(
            n_qubits=self.n_qubits, 
            t_vals=t,
            theta_params=self.theta_params,
            phi_params=self.phi_params,
            omega=self.omega,
            gamma=self.gamma,
            hbar=self.hbar,
            kraus_order=self.kraus_order,
            angle_tolerance=self.angle_tolerance)
        
        # convert initial state to numpy array:
        initial_state = np.array(initial_state).squeeze()
        initial_state_is_pure = (initial_state.ndim == 1)
        
        # build full density simulation circuit:
        density_kraus_circuits = []
        for kraus_circuit, conj_kraus_circuit in zip(self.kraus_circuits, self.conj_kraus_circuits):
            
            # create full density quantum circuit:
            n_left_qubits = len(kraus_circuit.qubits)
            n_right_qubits = len(conj_kraus_circuit.qubits)
            density_qc = QuantumCircuit(len(kraus_circuit.qubits)+len(conj_kraus_circuit.qubits))

            # determine indices of row/column qubits of simulated density matrix:
            #  Note: left qubits are acted on by kraus circuits, right qubits by conjugate kraus circuits:
            left_qubits = list(range(n_left_qubits))
            right_qubits = list(range(n_left_qubits, n_left_qubits+n_right_qubits))
            left_density_qubits = left_qubits[:self.n_qubits]
            right_density_qubits = left_qubits[:self.n_qubits]

            # add state prep gate to circuit:
            if initial_state_is_pure:
                density_qc.append(StatePreparation(initial_state.flatten()), left_density_qubits)
                density_qc.append(StatePreparation(initial_state.conj().flatten()), right_density_qubits)
            else:
                density_qc.append(StatePreparation(initial_state.flatten()),
                                  left_density_qubits+right_density_qubits)
            
            # add kraus circuits to simulate left and right time evolution:
            density_qc.append(kraus_circuit.decompose(), left_qubits)
            density_qc.append(conj_kraus_circuit.decompose(), right_qubits)

            density_kraus_circuits.append(density_qc)

        if magnitudes_only:
            pass
        else:
            pass

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
        
        rhos = simulate_lindblad_trajectory(initial_state, t,
                                           hamiltonian=self.hamiltonian_operator(),
                                           lindblad_ops=[ self.ladder_operator() ],
                                           gammas=[ self.gamma ],
                                           hbar=self.hbar)
        
        # compute expectation value of observable (if given):
        if observable is not None:
            return np.real(np.trace(observable @ rhos, axis1=1,axis2=2))
        
        return rhos

    def _plot_density_coutours(self, X, Y, Z, xlabel, ylabel, ax=None, levels=None, cmap='plasma'):
        new_plot = (ax is None)
        if new_plot:
            plt.figure()
            ax = plt.gca()
        
        if levels is None:
            cm = ax.pcolormesh(X,Y,Z, shading='gouraud', cmap=cmap, vmin=0)
        else:
            cm = ax.contourf(X,Y,Z, levels=levels, cmap=cmap, vmin=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if new_plot:
            plt.show()

    def plot_observable_density_trajectory(self, t, density_matrices, observable=None, observable_name=None, ax=None, cmap='plasma'):
    
        w, U = np.linalg.eigh(observable)
        U_dag = U.conj().T
        rho_w = np.array([ np.diag(U_dag @ rho @ U) for rho in density_matrices ])
        W, T = np.meshgrid(w, t)

        xlabel = r'$t$'
        ylabel= observable_name if observable_name is not None else 'Expectation Value'

        self._plot_density_coutours(X=T,Y=W,Z=np.real(rho_w), 
                                    xlabel=xlabel, 
                                    ylabel=ylabel, 
                                    ax=ax, 
                                    levels=100,
                                    cmap=cmap)

    def get_position_density_trajectory(self, density_matrices, x, m=1.0, unitless=False):
        rho_probs, rho_states = [], []
        for rho in density_matrices:
            w, U = np.linalg.eigh(rho)
            rho_probs.append(w)
            rho_states.append(U)

        x0_scale = np.sqrt(self.hbar / (m*self.omega))
        if unitless:
            x = x.copy() * x0_scale

        rho_probs = np.array(rho_probs)
        rho_states = np.array(rho_states)

        x_amplitudes = np.array([
            qho_eigenstate(n,x, omega=self.omega, m=m, hbar=self.hbar)
            for n in range(rho_states.shape[-1])
        ])
        U_amplitudes = np.einsum('ijk,jl->ikl', rho_states, x_amplitudes)
        densities = np.einsum('ik,ikl->il', rho_probs, np.abs(U_amplitudes)**2)

        return densities

    
    def plot_position_density_trajectory(self, t, density_matrices, m=1.0, ax=None, xlim=(-2,2), cmap='plasma', levels=100, unitless=False):
        
        x = np.linspace(*xlim, 1000)
        X, T = np.meshgrid(x, t)
        densities = self.get_position_density_trajectory(density_matrices=density_matrices, x=x, m=m, unitless=unitless)

        self._plot_density_coutours(X=T,Y=X,Z=densities, 
                                    xlabel=r'$t$', 
                                    ylabel=r'$X$', 
                                    ax=ax, 
                                    levels=levels,
                                    cmap=cmap)


    def get_momentum_density_trajectory(self, density_matrices, p, m=1.0, unitless=False):
        rho_probs, rho_states = [], []
        for rho in density_matrices:
            w, U = np.linalg.eigh(rho)
            rho_probs.append(w)
            rho_states.append(U)

        p0_scale = np.sqrt(self.hbar*m*self.omega)
        if unitless:
            p = p.copy()*p0_scale

        rho_probs = np.array(rho_probs)
        rho_states = np.array(rho_states)
        
        p_amplitudes = np.array([
            (-1.j)**n * qho_eigenstate(n,p, omega=1/(m*m*self.omega), m=m, hbar=self.hbar)
            for n in range(rho_states.shape[-1])
        ])
        U_amplitudes = np.einsum('ijk,jl->ikl', rho_states, p_amplitudes)
        densities = np.einsum('ik,ikl->il', rho_probs, np.abs(U_amplitudes)**2)

        return densities

    def plot_momentum_density_trajectory(self, t, density_matrices, m=1.0, ax=None, plim=(-2,2), cmap='plasma', levels=100, unitless=False):
        

        p = np.linspace(*plim, 1000)
        P, T = np.meshgrid(p, t)
        densities = self.get_momentum_density_trajectory(density_matrices=density_matrices, p=p, m=m, unitless=unitless)

        self._plot_density_coutours(X=T,Y=P,Z=densities,
                                    xlabel=r'$t$', 
                                    ylabel=r'$P$', 
                                    ax=ax, 
                                    levels=levels)


    def get_wigner_distribution_trajectory(self, density_matrices, x, p, m=1.0, tolerance=1e-5):

        X, P = np.meshgrid(x,p)
        W = np.zeros((density_matrices.shape[0],) + X.shape)
        g = np.sqrt(2)
        A = 0.5 * g * (X + 1.j*P)
        B = 4*np.abs(A)**2

        for i in range(density_matrices.shape[-2]):
            rho_ii = density_matrices[:,i,i]
            if np.any(np.abs(rho_ii) > tolerance):
                W_ii_basis = (-1)**i * genlaguerre(i,0)(B)
                W += np.real(np.einsum('i,jk->ijk', rho_ii, W_ii_basis))

            for j in range(i+1, density_matrices.shape[-1]):
                rho_ij = density_matrices[:,i,j]
                if np.any(np.abs(rho_ij) > tolerance):
                    W_ij_basis = 2.0 * (-1)**i * (2*A)**(j-i) * np.sqrt(factorial(i)/factorial(j)) * genlaguerre(i,j-i)(B)
                    W += np.real(np.einsum('i,jk->ijk', rho_ij, W_ij_basis))
                    

        return X, P, W *(g**2)*np.exp(-B/2)/ (2*np.pi)
    
    def plot_wigner_distribution_animation(self, t, density_matrices, m=1.0, xlim=(-2,2), plim=(-2,4), mesh_size=(100,100), 
                                        overlay=True, fig=None, ax=None, cmap='seismic', tolerance=1e-5, frame_interval=5):

        new_plot = (ax is None or fig is None)
        if new_plot:
            fig, ax = plt.subplots()

        x = np.linspace(*xlim, mesh_size[0])
        p = np.linspace(*plim, mesh_size[1])

        X, P, Z_t = self.get_wigner_distribution_trajectory(density_matrices, x, p, m, tolerance=tolerance)
        zlim = max(np.abs(np.min(Z_t)), np.abs(np.max(Z_t)))

        ax.grid(alpha=0.8)
        cax = ax.pcolormesh(X,P, Z_t[0], cmap=cmap, vmin=-zlim, vmax=zlim, shading='gouraud')
        timelabel = ax.text(0.9,0.9, "", transform=ax.transAxes, ha="right",
                            bbox=dict(boxstyle='round', facecolor='white'))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$p$')

        def _animate_wigner_distribution(frame):
            cax.set_array(Z_t[frame])
            timelabel.set_text(f't = {t[frame]:.3f}')

            return cax, timelabel

        anim = animation.FuncAnimation(fig, _animate_wigner_distribution, frames=len(t), interval=frame_interval)

        return anim

