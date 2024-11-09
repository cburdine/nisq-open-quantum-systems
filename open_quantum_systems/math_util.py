from itertools import product, islice

import numpy as np
from qiskit.quantum_info import SparsePauliOp

def comm(A,B):
    """Returns the commutator of A and B
    (i.e. $[A,B]$)

    Args:
        A (np.ndarray): A operator
        B (np.ndarray): B operator

    Returns:
        np.ndarray: Commutator
    """
    return A@B - B@A

def kron_sum(A_list):
    """Returns the Kronecker sum of a list of square matrix operators (as np.ndarrays)

    Args:
        A_list (list[np.ndarray]): List of operators

    Returns:
        np.ndarray: Kronecker sum of all operators.
    """
    A_dims = [ A.shape[-1] for A in A_list ]
    total = np.zeros((np.prod(A_dims), np.prod(A_dims)))
    for i, A in enumerate(A_list):
        dim_left = np.prod(A_dims[:i]).astype(np.int64)
        dim_right = np.prod(A_dims[i+1:]).astype(np.int64)
        A_i = np.kron(np.kron(np.eye(dim_left), A), np.eye(dim_right))
        total += A_i

    return total

def conj_kron2_sum(A):
    """Returns the conjugate kronecker sum 
    (i.e. $A \otimes I + I \otimes \overline{A}$)

    Args:
        A (np.ndarray): Operator

    Returns:
        np.ndarray: Conjugate kronecker sum
    """
    return np.kron(A,np.eye(A.shape[0])) + np.kron(np.eye(A.shape[0]),A.conj())

def conj_kron2(A):
    """Returns the conjugate kronecker product
    (i.e. $A \otimes \overline{A}$)

    Args:
        A (np.ndarray): Operator

    Returns:
        np.ndarray: Conjugate kronecker product
    """
    return np.kron(A,A.conj())

def zassenhaus_g(t=1, u=0, v=0):
    """Computes the Zassenhaus g(u,v,c) function used in the special case of the Zassenhaus 
    expansion formula
    $$exp(t(A + B)) = exp(tA)exp(tB) exp(g(t,u,v)*[A,B])$$

    where A and B satisfy the commutation relation $[A,B] = uA + vB + c$

    Args:
        t (float, np.ndarray): t values in Zassenhaus exponent
        u (float, optional): u coefficient. Defaults to 0.
        v (float, optional): v coefficient. Defaults to 0.
    
    Returns:
        float: Value of g(u,v,t)
    """

    if u == v == 0:
        return -0.5*(t**2)*np.ones(np.broadcast(u,v))
    elif u == 0:
        return -(np.exp(-t*v) - 1 + t*v)/v**2
    elif v == 0:
        return -(np.exp(t*u)*(1-t*u)-1)/u**2
    elif u == v:
        return ((t*u) + 1 - np.exp(u*t))/u**2
    else:
        return (u*np.exp(-t*(u-v)) + v*(np.exp(t*u)-1))/(u*v*(u-v))

def get_pauli_strings(n, bases='IZXY', limit=None):
    """Returns a complete list of n-qubits Pauli strings in the indicated bases and order.
       A limiting number of Pauli strings my be specified. Note that if no limit is set,
       a total of 4**n Pauli strings will be generated.

    Args:
        n (int): Number of qubits
        bases (str, optional): Basis string consisting of ordered Pauli operators. Defaults to 'IZXY'.
        limit (int, optional): Maximum number of Pauli strings to generate. Defaults to None.

    Returns:
        list[str]: List of pauli strings.
    """
    if limit is None:
        return list(enumerate_pauli_strings(n,bases))
    
    return islice(enumerate_pauli_strings(n,bases),limit)

def enumerate_pauli_strings(n, bases='IZXY'):
    """Enumerates n-qubits Pauli strings in the indicated bases and order.

    Args:
        n (int): Number of qubits
        bases (str, optional): Basis string consisting of ordered Pauli operators. Defaults to 'IZXY'.

    Yields:
        str: The next pauli string.
    """
    basis_strings = [bases]*n
    for s in product(*basis_strings):
        yield ''.join(s)

def enumerate_subbasis_strings(basis_str):
    """Enumerates all sub-basis strings of a given Pauli basis string

    Args:
        basis_str (str): A basis measurement Pauli string (cannot contain any I's)

    Yields:
        str: A subbasis string of the given measurement basis
    """
    assert 'I' not in basis_str
    for mask in range(1<<len(basis_str)):
        subbasis_str = [
            ch if mask&(1<<i) else 'I'
            for i, ch in enumerate(basis_str)
        ]
        yield ''.join(subbasis_str)


def make_pauli_operator(pauli_str):
    """Generates a Pauli operator in matrix form from a Pauli string.

    Args:
        pauli_str (str): A pauli string. The length of the Pauli string determines the operator rank.

    Raises:
        ValueError: If a Pauli character is invalid (must be from 'I', 'X', 'Y', or 'Z').

    Returns:
        np.ndarray: Operator in matrix form as a 2D numpy array.
    """
    if isinstance(pauli_str, dict):
        return make_operator_from_pauli_dict
    
    if len(pauli_str) == 0:
        return np.array([[1]])
    elif len(pauli_str) == 1:
        if pauli_str == 'I':
            return np.eye(2)
        if pauli_str == 'X':
            return np.array([[0,1],[1,0]])
        elif pauli_str == 'Y':
            return np.array([[0,-1.j],[1.j,0]])
        elif pauli_str == 'Z':
            return np.array([[1, 0],[0,-1]])
        else:
            raise ValueError(f'Unknown Pauli Character: {pauli_str}')
    else:
        left_str = pauli_str[:len(pauli_str)//2]
        right_str = pauli_str[len(pauli_str)//2:]
        return np.kron(make_pauli_operator(left_str),make_pauli_operator(right_str))

def to_pauli_dict(observable, paulis='IZXY', tol=1e-9):
    """Decomposes an observable in matrix form as a dictionary of Pauli strings and associated weight value.

    Args:
        observable (_type_): _description_
        paulis (str, optional): Pauli string bases. Defaults to 'IZXY'.
        tol (float, optional): Tolerance of decomposition. If the weight of a 
            given string is below the tolerance level, the string is excluded. Defaults to 1e-9.

    Returns:
        dict[str,float]: Dictionary of pauli strings -> weights.
    """
    n = int(np.ceil(np.log2(observable.shape[0])))
    pauli_dict = {}
    for pauli_str in enumerate_pauli_strings(n, paulis):
        pauli_op = make_pauli_operator(pauli_str).flatten()
        x = np.dot(pauli_op.flatten().conj(), observable.flatten())/2**n
        if np.abs(x) > tol:
            pauli_dict[pauli_str] = np.real_if_close(x)

    return pauli_dict

def make_operator_from_pauli_dict(pauli_dict):
    """Constructs an operator in matrix form a pauli string dictionary

    Args:
        pauli_dict (dict): Dictionary mapping Pauli strings to their corresponding weights.

    Returns:
        np.ndarray: Constructed operator in matrix form as a 2D numpy array.
    """
    n = max(len(s) for s in pauli_dict.keys())
    sample_dict_val = np.array(np.array(next(iter(pauli_dict.values()))) if pauli_dict else 0)
    op = np.zeros((2**n,2**n) + sample_dict_val.shape,
                    dtype=np.complex128)
    for s, x in pauli_dict.items():
        # pad pauli string on the left if needed:
        if len(s) < n:
            s = ('I'*(n-len(s)) + s)
        pauli_op = make_pauli_operator(s)

        # make pauli op and x shapes compatible:
        if not np.isscalar(x):
            pauli_op = pauli_op.reshape(pauli_op.shape + (1,)*sample_dict_val.ndim)
            x = x.reshape((1,1) + sample_dict_val.shape)
        op += pauli_op*x
        
    return np.real_if_close(op)

def pauli_commutator(a, b, a_coeff=1., b_coeff=1.):
    
    PAULI_CHARS = 'XYZI'
    ord_a = PAULI_CHARS.index(a)
    ord_b = PAULI_CHARS.index(b)

    if ord_a == 3 or ord_b == 3 or ord_a == ord_b:
        return 'I', 0

    eps_ijk = (1 if (ord_b-ord_a)%3 == 1 else -1)
    pauli_ch = PAULI_CHARS[(ord_b+eps_ijk)%3]
    a = 2.j*eps_ijk*a_coeff*b_coeff

    return a, pauli_ch

def pauli_anticommutator(a, b, a_coeff=1., b_coeff=1.):
    
    PAULI_CHARS = 'XYZI'
    ord_a = PAULI_CHARS.index(a)
    ord_b = PAULI_CHARS.index(b)

    if ord_a == ord_b:
        return (2., a)
    elif ord_a == 3:
        return (2., ord_b)
    elif ord_b == 3:
        return (2., ord_a)
    else:
        return 0

def pauli_string_commutator(str_a, str_b, a_coeff=1., b_coeff=1.):
    
    assert(len(str_a) == len(str_b))

    # iterate character-by-character:
    noncomm_subspaces = []
    for i, (ch_a, ch_b) in enumerate(zip(str_a, str_b)):
        if ch_a != ch_b and ch_a != 'I' and ch_b != 'I':
            noncomm_subspaces.append(i)

    if len(noncomm_subspaces) % 2 == 0:
        return (0, 'I'*len(str_a))
    
def common_measurement_basis(pauli_list, default='Z'):
    
    assert(len(default) == 1)
    basis = ['I']*len(pauli_list[0])
    for pauli in pauli_list:
        assert(len(pauli) == len(basis))
        for i, ch in enumerate(pauli):
            if ch != 'I' and ch != basis[i]:
                assert(basis[i] == 'I')
                basis[i] = ch
    basis_str = ''.join(basis).replace('I', default)
    return basis_str

def is_qubitwise_subbasis(pauli, basis):

    assert(len(pauli) == len(basis))
    for ch_p, ch_b in zip(pauli, basis):
        if ch_p != 'I' and ch_p != ch_b:
            return False
    return True

def group_commuting_pauli_dicts(pauli_dict, qubit_wise=True):
    sparse_pauli = SparsePauliOp.from_list(list(pauli_dict.items()))
    commuting_pauli_dicts = []
    for sparse_pauli in sparse_pauli.group_commuting(qubit_wise):
        commuting_pauli_dicts.append(dict(sparse_pauli.to_list()))

    return commuting_pauli_dicts

def get_measurement_basis_pauli_dicts(pauli_dict):
    
    pauli_dict_groups = group_commuting_pauli_dicts(pauli_dict, qubit_wise=True)
    bases = [
        common_measurement_basis(list(pd.keys()))
        for pd in pauli_dict_groups
    ]

    # construct map of strs -> compatible measurement bases:
    str_basis_map = {
        s : [b for b in bases if is_qubitwise_subbasis(s, b)]
        for s in pauli_dict
    }
    
    # distribute the weights of the pauli dict entries
    # evenly among compatible measurement bases:
    measurement_basis_dicts = { b : {} for b in bases }
    for s, bases in str_basis_map.items():
        basis_x = (pauli_dict[s] / len(bases))
        for b in bases:
            measurement_basis_dicts[b][s] = basis_x

    return measurement_basis_dicts
                