import numpy as np
import os
from scipy.linalg import expm
import networkx as nx



def is_acyclic(mask):
    G = nx.DiGraph()
    n_vars = mask.shape[0]
    for i in range(n_vars):
        for j in range(n_vars):
            if mask[i, j] != 0:
                G.add_edge(j, i)  # cause -> effect
    return mask.shape[0] == np.matrix.trace(expm(mask))


def select_unfaithful_connections(shape,n_unfaithful):
    cause_selection = np.arange(shape[0])
    effect_selection = np.arange(shape[0])
    indirect_connection = np.arange(shape[0])
    while (
        (cause_selection == effect_selection).any()
        or (cause_selection == indirect_connection).any()
        or (effect_selection == indirect_connection).any()
    ):
        np.random.shuffle(cause_selection)
        np.random.shuffle(effect_selection)
        np.random.shuffle(indirect_connection)
    # selection 3 vars to form a unfaithful connection
    cause_selection = cause_selection[:n_unfaithful]
    effect_selection = effect_selection[:n_unfaithful]
    indirect_connection = indirect_connection[:n_unfaithful]
    # Cause, Effect, Indirect
    connections = np.stack(
        [cause_selection, effect_selection, indirect_connection], axis=0
    ).T
    # assign with increased number of distortions.
    return connections


def unfaithful_instant_masks(
    n_samples=100,
    n_unfaithful=2,
    shape=(5, 5),
    distortion=[0.1, 0.05, 0.025, 0.01, 0],
):
    """
    Builds full stack of violation masks
    """
    outer_stack=[]
    for x in range(n_samples):
        
        proper = False
        while not proper:
            connections = select_unfaithful_connections(shape,n_unfaithful)
            potential = np.zeros(shape)
            for con in connections:
                potential[con[1], con[0]] = 1
                potential[con[2], con[0]] = 1# add distortion to be not fully unfaithfull
                potential[con[1], con[2]] = 1
            if is_acyclic(potential):
                # Checks whether all unfaithful connections are fully seperated 
                # (otherwise this overcharges the parameterization.)
                if (potential != 0).sum() == (n_unfaithful *3):
                    proper = True
            else:
                print("resampling")
       
        stack = []
        for dist in distortion:
            potential = np.zeros(shape)
            for con in connections:
                val = np.random.uniform(0.3,0.5)
                potential[con[1], con[0]] += (-val + dist)
                potential[con[2], con[0]] += (2 * val)  # add distortion to be not fully unfaithfull
                # keep the lower boundary of the connection to be above the 0.3 (typical lower bound.)
                potential[con[1], con[2]] += 0.5 
                # I guess this work? TODO CHECK via COnd indepedence.
            stack.append(potential)
        outer_stack.append(stack)
    return np.swapaxes(np.array(outer_stack), 0,1)


def check_var_stability(coefficients):
    """
    Checks the stability of a VAR(p) process.
    Args:
        A_matrices (np.ndarray): A 3D numpy array of shape (p, k, k)
                                containing the coefficient matrices A_1, ..., A_p.
                                p = number of lags
                                k = number of time series
    Returns:
        tuple: A tuple containing:
            - bool: True if the process is stable, False otherwise.
            - float: The maximum eigenvalue modulus.
    """
    A_matrices = coefficients.T
    if A_matrices.ndim != 3 or A_matrices.shape[1] != A_matrices.shape[2]:
        raise ValueError("A_matrices must be a 3D array of shape (p, k, k).")

    p, k, _ = A_matrices.shape
    # The total dimension of the companion matrix
    kp = k * p
    # Construct the companion matrix F
    companion_matrix = np.zeros((kp, kp))
    # Place A_1, ..., A_p in the first k rows
    companion_matrix[0:k, :] = np.hstack(A_matrices)
    # Place the identity matrix in the lower-left block
    if p > 1:
        identity_block = np.eye(k * (p - 1))
        companion_matrix[k:, 0:k*(p-1)] = identity_block
    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(companion_matrix)
    # Calculate the modulus (absolute value) of the eigenvalues
    eigenvalue_moduli = np.abs(eigenvalues)
    # Find the maximum modulus
    max_modulus = np.max(eigenvalue_moduli)

    return max_modulus < 1.0, max_modulus


def unfaithful_lagged_masks(
    n_samples=100,
    n_unfaithful=2,
    shape=(5, 5,3),
    distortion=[0.4,0.1, 0.05, 0.025, 0.01, 0],
):
    """
    Builds full stack of violation masks
    """
    outer_stack=[]
    resamples = 0 
    double_count = 0
    while len(outer_stack) < n_samples:
        proper = False
        while not proper:
            connections = select_unfaithful_connections(shape,n_unfaithful)       
            direct_cause_lag = 1
            indirect_lag_1 = 0
            indirect_lag_2 = 0
            potential = np.zeros(shape)
            for con in connections:
                potential[con[1], con[0],direct_cause_lag] = 1
                potential[con[2], con[0],indirect_lag_1] = 1
                potential[con[1], con[2],indirect_lag_2] = 1
                # if this doesnt match we have double links.
            if (potential != 0).sum() == (n_unfaithful *3):
                proper = True
            else:
                double_count += 1
            
        stack = []
        for dist in distortion:
            potential = np.zeros(shape)
            for con in connections:
                val = np.random.uniform(0.3,0.5)
                # This is weird to restrict as we need to also generate things that dont diverge.
                potential[con[1], con[0],direct_cause_lag] += (-val + dist)
                potential[con[2], con[0],indirect_lag_1] +=(2 * val)   # add distortion to be not fully unfaithfull
                potential[con[1], con[2],indirect_lag_2] += (0.5)

            stack.append(potential)
        if np.all([check_var_stability(x)[0] for x in stack]):
            outer_stack.append(stack)
        else:
            resamples+=1
    print("Resampling due to divergence. Resamples: ", resamples)
    print("resampling due to double links. Count: ", double_count)
    return np.swapaxes(np.array(outer_stack), 0,1)


def export_lagged_faithful(distortion=0, n_unfaithful=8, shape=(7,7,4)):
    masks = unfaithful_lagged_masks(distortion= distortion,n_unfaithful=n_unfaithful, shape=shape)
    print(np.abs(masks[masks != 0]).min(), np.abs(masks[masks != 0]).max())
    for n,x in enumerate(distortion):
        path = "../masks/masks_" + str(shape[0]) + "_" + str(n_unfaithful) + "_faith_lagged_" + str(4-n) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        for m,item in enumerate(masks[n]):
            np.save(path + str(m) + ".npy", item)