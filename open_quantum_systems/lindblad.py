import numpy as np
from scipy.integrate import solve_ivp

def simulate_lindblad_trajectory(initial_state, t, hamiltonian, lindblad_ops=None, gammas=None, hbar=1.):

    # get Hamiltonian and determine system rank:
    H = hamiltonian
    rank = hamiltonian.shape[0]
    
    # construct lindblad operators (if given):
    nonunitary = len(lindblad_ops) > 0
    if nonunitary:
        L = np.array(lindblad_ops)
        L_dagger = L.conj().transpose(0,2,1)
        L_squared = L_dagger @ L
        g = np.array(gammas).reshape(-1,1,1)

    # generate righthand side of Lindblad equation:
    def _lindblad_rhs(t, rho):
        rho = rho.reshape(rank,rank)
        rho_dot = (-1.j/hbar)*(H @ rho - rho @ H)
        if nonunitary:
                rho_dot += np.sum(
                    g*(L @ rho @ L_dagger + \
                    -0.5*(L_squared @ rho + rho @ L_squared)),
                    axis=0)

        return rho_dot.flatten()

    # create initial density matrix:
    initial_state = np.array(initial_state).squeeze()
    if initial_state.ndim == 1:
        assert(initial_state.size == rank)
        rho_init = np.outer(initial_state, initial_state.conj())
    else:
        assert(initial_state.shape[0] == rank)
        rho_init = initial_state

    # create system and integrate it:
    soln = solve_ivp(_lindblad_rhs,
            t_span=(t[0], t[-1]),
            y0=rho_init.flatten(),
            t_eval=t)

    rho_trajectory = soln.y.transpose().reshape(-1,rank,rank)

    return rho_trajectory

        