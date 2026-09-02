import numpy as np
import scipy as sp
import networkx as nx


def AIRM(A:np.ndarray, B: np.ndarray)->float:
    """
    Computes the Affine Invariant Riemannian Metric (AIRM) of two Symmetric Positive Definite (SPD) matrices A and B. 

    The function uses the generalized eigenvalue problem to compute the AIRM.

    Note that for efficiency reasons it is not checked if the inputs are positive definite. 
    
    --------------------------------------------------------------------------------------

    Parameters:

    A (np.array): First matrix

    B (np.array): Second matrix
    
    Returns:

    float: AIRM of A and B. 
    """
    assert isinstance(A, np.ndarray), "A should be a numpy array"
    assert isinstance(B, np.ndarray), "B should be a numpy array"
    assert A.shape == B.shape, "A and B should have the same shape"
    assert len(A.shape) == 2 and len(A.shape) == 2, "A and B should be 2D matrices"
    assert A.shape[0] == A.shape[1], "A should be a square matrix"
    assert B.shape[0] == B.shape[1], "B should be a square matrix"
    assert np.allclose(A, A.T), "A should be a symmetric matrix"
    assert np.allclose(B, B.T), "B should be a symmetric matrix"

    
    def get_eigenvalues(A,B):
        """
        Gets generalized eigenvalues of A and B.
        """
        return sp.linalg.eigh(A, B, eigvals_only=True)
    

    return np.sqrt(np.sum(np.power(np.log(get_eigenvalues(A,B)),2))) 


def get_K_eigenbasis_matrix(N: int) -> np.ndarray:
    """
    Creates a orthogonal matrix with ON eigenvectors of a complete graph with N nodes.
    
    Parameters:

    N (int): Number of nodes
    
    Returns:

    np.array: Orthogonal matrix with ON eigenvectors of a complete graph with N nodes. 
    
    """
    assert isinstance(N, int), "N should be an integer"
    assert N > 0, "N should be a positive integer"

    S = np.zeros((N, N))
    S[:, 0] = 1
    S[:, 0] = normalize_vector(S[:, 0])

    for i in range(N-1):
        S[i, i+1] = i+1-N
        S[i+1:, i+1] = 1
        S[:, i+1] = normalize_vector(S[:, i+1])

    return S


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector to have unit length. 
    Zero vector is returned as itself. 
    
    Parameters:

    vec (np.array): Vector to be normalized
    
    Returns:

    np.array: Normalized vector
    
    """
    assert isinstance(vec, np.ndarray), "vec should be a numpy array"
    assert len(vec.shape) == 1, "vec should be a 1D array"
    
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def get_connected_laplacian(G: nx.Graph)->np.ndarray:
    """
    Computes laplacian matrix of a connected graph. 
    
    Parameters:

    G (nx.Graph): Connected graph to compute the laplacian of 
    
    Returns:

    np.array: laplacian matrix of the graph
    
    """
    assert isinstance(G, nx.Graph), "G should be a networkx graph"
    assert G.number_of_nodes() > 0, "G should have at least one node"
    assert G.number_of_edges() > 0, "G should have at least one edge"
    assert nx.is_connected(G), "G should be a connected graph"

    return nx.laplacian_matrix(G).toarray()


def get_SPD_laplacian_submatrix(Q: np.ndarray) -> np.ndarray:
    """
    Calculates positive definite part of a laplacian matrix in the eigenbasis of the complete graph.

    Parameters:

    Q (np.array): laplacian matrix in the node basis

    Returns:

    np.array: positive-definite part of the laplacian matrix in the eigenbasis of the compete graph 
    """

    assert isinstance(Q, np.ndarray), "Q should be a numpy array"
    assert len(Q.shape) == 2, "Q should be a 2D matrix"
    assert Q.shape[0] == Q.shape[1], "Q should be a square matrix"
    assert np.array_equal(Q, Q.T), "Q should be a symmetric matrix"

    return similarity_by_othogonal_matrix(Q, get_K_eigenbasis_matrix(Q.shape[0]))[1:,1:]


def similarity_by_othogonal_matrix(A: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Calulates a simirality transform of the matrix A by an orthogonal matrix S. 

    Parameters: 

    A (np.array): matrix to be transformed
    S (np.array): orthogonal matrix to transform A

    Returns:
    np.array: S.T @ A S transformed matrix
    """

    assert isinstance(A, np.ndarray), "A should be a numpy array"
    assert isinstance(S, np.ndarray), "S should be a numpy array"
    assert A.shape == S.shape, "A and S should have the same shape"
    assert len(A.shape) == 2, "A should be a 2D matrix"
    assert len(S.shape) == 2, "S should be a 2D matrix"
    assert A.shape[0] == A.shape[1], "A should be a square matrix"
    assert S.shape[0] == S.shape[1], "S should be a square matrix"
    assert np.allclose(S.T @ S, np.eye(S.shape[0])), "S should be an orthogonal matrix"
    assert np.allclose(S @ S.T, np.eye(S.shape[0])), "S should be an orthogonal matrix"

    return S.T @ A @ S


def GGD(G1: nx.graph, G2: nx.graph) -> float:
    """
    Calculates the Graph Geodesic Distance(GGD) between two graphs G1 and G2. 

    Parameters:

    G1 (nx.Graph): First graph
    G2 (nx.Graph): Second graph

    Returns:

    float: GGD between G1 and G2
    """

    assert isinstance(G1, nx.Graph), "G1 should be a networkx graph"
    assert isinstance(G2, nx.Graph), "G2 should be a networkx graph"
    assert G1.number_of_nodes() > 0, "G1 should have at least one node"
    assert G2.number_of_nodes() > 0, "G2 should have at least one node"
    assert G1.number_of_edges() > 0, "G1 should have at least one edge"
    assert G2.number_of_edges() > 0, "G2 should have at least one edge"
    assert nx.is_connected(G1), "G1 should be a connected graph"
    assert nx.is_connected(G2), "G2 should be a connected graph"
    assert G1.number_of_nodes() == G2.number_of_nodes(), "G1 and G2 should have the same number of nodes"

    return AIRM(get_SPD_laplacian_submatrix(get_connected_laplacian(G1)), 
                get_SPD_laplacian_submatrix(get_connected_laplacian(G2)))


def GGD_normalized(G1: nx.graph, G2: nx.graph) -> float:
    """
    Calculates the Graph Geodesic Distance(GGD) normalized by sqrt(number_of_nodes-1) between two graphs G1 and G2. 

    Parameters:

    G1 (nx.Graph): First graph
    G2 (nx.Graph): Second graph

    Returns:

    float: GGD/sqrt(N-1) between G1 and G2
    """
    return GGD(G1, G2)/np.sqrt(G1.number_of_nodes()-1)   


def make_random_permutation(N: int) -> np.ndarray:
    """
    Creates a random permutation vector of size N.
    
    Parameters:

    N (int): Size of the permutation vector
    
    Returns:

    np.array: Random permutation vector of size N
    
    """
    assert isinstance(N, int), "N should be an integer"
    assert N > 0, "N should be a positive integer"

    return np.random.permutation(N)
    

def compose_permutations(perm1: np.ndarray, perm2: np.ndarray) -> np.ndarray:
    """
    Composes two permutation vectors.
    !VIBECODED!

    Parameters:
    
    perm1 (np.array): First permutation vector
    perm2 (np.array): Second permutation vector
    
    Returns:
    
    np.array: Composition of perm1 and perm2, i.e. perm1(perm2(i))
    """
    assert isinstance(perm1, np.ndarray), "perm1 should be a numpy array"
    assert isinstance(perm2, np.ndarray), "perm2 should be a numpy array"
    assert len(perm1.shape) == 1, "perm1 should be a 1D array"
    assert len(perm2.shape) == 1, "perm2 should be a 1D array"
    assert np.issubdtype(perm1.dtype, np.integer), "perm1 should be an integer array"
    assert np.issubdtype(perm2.dtype, np.integer), "perm2 should be an integer array"
    assert np.all(np.isin(perm1, np.arange(len(perm1)))), "perm1 should be a permutation of (0,1,2...N-1)"
    assert np.all(np.isin(perm2, np.arange(len(perm2)))), "perm2 should be a permutation of (0,1,2...N-1)"
    assert len(perm1) == len(perm2), "perm1 and perm2 should have the same length"

    return perm1[perm2]


def inverse_permutation(perm:np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a permutation vector.
    !VIBECODED!

    Parameters:
    
    perm (np.array): Permutation vector to be inverted
    
    Returns:
    
    np.array: Inverse permutation vector
    """
    assert isinstance(perm, np.ndarray), "perm should be a numpy array"
    assert len(perm.shape) == 1, "perm should be a 1D array"
    assert np.issubdtype(perm.dtype, np.integer), "perm should be an integer array"
    assert np.all(np.isin(perm, np.arange(len(perm)))), "perm should be a permutation of (0,1,2...N-1)"

    inv_perm = np.zeros_like(perm)
    for i in range(len(perm)):
        inv_perm[perm[i]] = i
    return inv_perm
    

def similarity_transform_permutation(A, perm):
    """
    Efficiently compute PAP^T where P is a permutation matrix.
    !VIBECODED!
    
    Parameters:
    -----------
    A : np.ndarray
        Square matrix (n x n)
    perm : np.ndarray
        Permutation vector where perm[i] = j means row i -> position j
        (or row_assignments from greedy algorithm)
    
    Returns:
    --------
    result : np.ndarray
        Transformed matrix PAP^T
    """
    # Apply permutation to rows, then to columns
    # This is equivalent to PAP^T
    return A[perm,:][:,perm]
    

def GGD_with_relabeling(G1: nx.graph, G2: nx.graph, perm: np.ndarray) -> float:
    """
    Calculates the Graph Geodesic Distance(GGD) between two graphs G1 and G2. With relabeling of G2 by permutation matrix P. 

    Parameters:

    G1 (nx.Graph): First graph
    G2 (nx.Graph): Second graph
    P (np.array): permutation to relabel G2

    Returns:

    float: GGD between G1 and P.G2.P^T  
    """

    assert isinstance(G1, nx.Graph), "G1 should be a networkx graph"
    assert isinstance(G2, nx.Graph), "G2 should be a networkx graph"
    assert G1.number_of_nodes() > 0, "G1 should have at least one node"
    assert G2.number_of_nodes() > 0, "G2 should have at least one node"
    assert G1.number_of_edges() > 0, "G1 should have at least one edge"
    assert G2.number_of_edges() > 0, "G2 should have at least one edge"
    assert nx.is_connected(G1), "G1 should be a connected graph"
    assert nx.is_connected(G2), "G2 should be a connected graph"
    assert G1.number_of_nodes() == G2.number_of_nodes(), "G1 and G2 should have the same number of nodes"
    assert isinstance(perm, np.ndarray), "P should be a numpy array"
    assert len(perm.shape) == 1, "P should be a 1D array"
    assert perm.shape[0] == G2.number_of_nodes(), "P should have the same size as the number of nodes in G2"
    assert np.issubdtype(perm.dtype, np.integer), "P should be an integer array"
    assert np.all(np.isin(perm, np.arange(G2.number_of_nodes()))), "P should be a permutation of (0,1,2...N-1)"

    return AIRM(get_SPD_laplacian_submatrix(get_connected_laplacian(G1)), 
                get_SPD_laplacian_submatrix(similarity_transform_permutation(get_connected_laplacian(G2), perm)))


def GGD_normalized_with_relabeling(G1: nx.graph, G2: nx.graph, perm: np.ndarray) -> float:
    """
    Calculates the Graph Geodesic Distance(GGD) normalized by sqrt(number_of_nodes-1) between two graphs G1 and G2. 
    With relabeling of G2 by permutation p. 

    Parameters:

    G1 (nx.Graph): First graph
    G2 (nx.Graph): Second graph
    P (np.array): permutation to relabel G2

    Returns:

    float: GGD/sqrt(N-1) between G1 and P.G2.P^T  
    """
    return GGD_with_relabeling(G1, G2, perm)/np.sqrt(G1.number_of_nodes()-1)

  
def GGD_bounds(G1: nx.graph, G2: nx.graph) -> tuple:
    """
    Calculates lower and upper bounds for the Graph Geodesic Distance(GGD) between two graphs G1 and G2. 

    Parameters:

    G1 (nx.Graph): First graph
    G2 (nx.Graph): Second graph

    Returns:

    tuple: (float: lower bound, float: upper bound) for GGD between G1 and G2
    """

    assert isinstance(G1, nx.Graph), "G1 should be a networkx graph"
    assert isinstance(G2, nx.Graph), "G2 should be a networkx graph"
    assert G1.number_of_nodes() > 0, "G1 should have at least one node"
    assert G2.number_of_nodes() > 0, "G2 should have at least one node"
    assert G1.number_of_edges() > 0, "G1 should have at least one edge"
    assert G2.number_of_edges() > 0, "G2 should have at least one edge"
    assert nx.is_connected(G1), "G1 should be a connected graph"
    assert nx.is_connected(G2), "G2 should be a connected graph"
    assert G1.number_of_nodes() == G2.number_of_nodes(), "G1 and G2 should have the same number of nodes"

    Q1 = get_connected_laplacian(G1)
    Q2 = get_connected_laplacian(G2)

    eigs_Q1 = np.linalg.eigvalsh(Q1)
    eigs_Q2 = np.linalg.eigvalsh(Q2)

    lower_bound = np.sqrt(np.sum(np.power(np.log(np.sort(eigs_Q1)[1:]/np.sort(eigs_Q2)[1:]),2)))
    upper_bound = np.sqrt(np.sum(np.power(np.log(np.sort(eigs_Q1)[1:]/np.sort(eigs_Q2)[1:][::-1]),2)))

    return (lower_bound, upper_bound)


def GGD_normalized_bounds(G1: nx.graph, G2: nx.graph) -> tuple:
    """
    Calculates lower and upper bounds for the Graph Geodesic Distance(GGD) normalized by sqrt(N-1) between two graphs G1 and G2. 

    Parameters:

    G1 (nx.Graph): First graph
    G2 (nx.Graph): Second graph

    Returns:

    tuple: (float: lower bound, float: upper bound) for GGD normalized by sqrt(N-1) between G1 and G2
    """
    return (GGD_bounds(G1, G2)[0]/np.sqrt(G1.number_of_nodes()-1), GGD_bounds(G1, G2)[1]/np.sqrt(G1.number_of_nodes()-1))


def make_GRAMPA_similarity_matrix(A:np.ndarray, B:np.ndarray, bandwith:float) -> np.ndarray:
    """
    Creates a similarity matrix of matrices A and B for GRAMPA algorithm to be further rounded to a permutation matrix.   
    
    Parameters:

    A (np.array): Adjacency matrix or normalized adjacency matrix of the first graph
    B (np.array): Adjacency matrix or normalized adjacency matrix of the second graph
    bandwith (float): Bandwith parameter for the Cauchy kernel
    
    Returns:

    np.array: Similarity matrix for GRAMPA algorithm to be rounded to a permutation matrix.
    
    VERY INEFFICIENT!!!
    """

    assert isinstance(A, np.ndarray), "A should be a numpy array"
    assert isinstance(B, np.ndarray), "B should be a numpy array"
    assert len(A.shape) == 2, "A should be a 2D matrix"
    assert len(B.shape) == 2, "B should be a 2D matrix"
    assert A.shape[0] == A.shape[1], "A should be a square matrix"
    assert B.shape[0] == B.shape[1], "B should be a square matrix"
    assert A.shape[0] == B.shape[0], "A and B should have the same size"
    assert isinstance(bandwith, float), "bandwith should be a float"
    assert bandwith > 0, "bandwith should be positive"
    assert np.allclose(A, A.T), "A should be a symmetric matrix"
    assert np.allclose(B, B.T), "B should be a symmetric matrix"

    N = A.shape[0]
   
    def cauchy_kernel(x:float, y:float, bandwith:float) -> float:
        """
        Cauchy kernel function.
        """
        return 1/(bandwith**2 + (x - y)**2)
    
    eigenvalues_A, eigenvectors_A = np.linalg.eigh(A)
    eigenvalues_B, eigenvectors_B = np.linalg.eigh(B)

    X = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            X += cauchy_kernel(eigenvalues_A[i], eigenvalues_B[j], bandwith) \
                * np.outer(eigenvectors_A[:,i], eigenvectors_B[:,j]) * np.sum(eigenvectors_A[:,i]) * np.sum(eigenvectors_B[:,j]) 

    return X
    
    
def make_GRAMPA_similarity_matrix_vectorized(A: np.ndarray, B: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Creates a similarity matrix of matrices A and B for GRAMPA algorithm to be further rounded to a permutation matrix.   
    
    Parameters:

    A (np.array): Adjacency matrix or normalized adjacency matrix of the first graph
    B (np.array): Adjacency matrix or normalized adjacency matrix of the second graph
    bandwith (float): Bandwith parameter for the Cauchy kernel
    
    Returns:

    np.array: Similarity matrix for GRAMPA algorithm to be rounded to a permutation matrix.
    
    !VIBECODED!
    """

    assert isinstance(A, np.ndarray), "A should be a numpy array"
    assert isinstance(B, np.ndarray), "B should be a numpy array"
    assert len(A.shape) == 2, "A should be a 2D matrix"
    assert len(B.shape) == 2, "B should be a 2D matrix"
    assert A.shape[0] == A.shape[1], "A should be a square matrix"
    assert B.shape[0] == B.shape[1], "B should be a square matrix"
    assert A.shape[0] == B.shape[0], "A and B should have the same size"
    assert isinstance(bandwidth, float), "bandwith should be a float"
    assert bandwidth > 0, "bandwith should be positive"
    assert np.allclose(A, A.T), "A should be a symmetric matrix"
    assert np.allclose(B, B.T), "B should be a symmetric matrix"

    def cauchy_kernel_matrix(evals_A, evals_B, bandwidth):
        # evals_A, evals_B: shape (N,)
        diff = evals_A[:, None] - evals_B[None, :]      # (N, N)
        return 1.0 / (bandwidth**2 + diff**2)           # (N, N)

        N = A.shape[0]

    # Eigen-decompositions
    evals_A, evecs_A = np.linalg.eigh(A)            # (N,), (N, N)
    evals_B, evecs_B = np.linalg.eigh(B)            # (N,), (N, N)

    # Cauchy kernel on eigenvalues
    K = cauchy_kernel_matrix(evals_A, evals_B, bandwidth)   # (N, N)

    # Sums of eigenvectors (per eigenvector)
    sA = evecs_A.sum(axis=0)                        # (N,)
    sB = evecs_B.sum(axis=0)                        # (N,)

    # Combine scalar weights into a single weight matrix W_ij
    W = K * (sA[:, None] * sB[None, :])             # (N, N)

    # Vectorized sum over outer products:
    # X_{pq} = sum_{i,j} W_ij * evecs_A[p,i] * evecs_B[q,j]
    # This is a double contraction that can be done with einsum.
    X = np.einsum('ij,pi,qj->pq', W, evecs_A, evecs_B, optimize=True)  # (N, N)

    return X


def hungarian_algorithm(X: np.ndarray) -> np.ndarray:
    """
    Given a matrix X (n x n),
    return a permutation p as a 1D array of length n such that
    p[i] is the column index assigned to row i, maximizing sum |X[i, p[i]]|.
    !!!VIBECODED!!!
    """
    row_ind, col_ind = sp.optimize.linear_sum_assignment(-X)  # Hungarian algorithm[web:31]

    # row_ind is sorted, so this is already in row order
    # permutation p: maps row i -> column p[i]

    return col_ind
    

def make_GRAMPA_permutation(G1:nx.graph, G2:nx.graph, bandwidth:float)-> np.ndarray:
    """
    Makes the GRAMPA permutation matrix to align graph G2 to the graph G1 based on their adjacency matrices.
    Using Hungarian algorithm for rounding the similarity matrix. 


    Parameters:

    G1 (nx.Graph): First graph
    G2 (nx.Graph): Second graph
    bandwith (float): Bandwith parameter for the Cauchy kernel

    Returns:

    np.array: Permutation matrix to align G2 to G1 based on their adjacency matrices.
    """

    assert isinstance(G1, nx.Graph), "G1 should be a networkx graph"
    assert isinstance(G2, nx.Graph), "G2 should be a networkx graph"
    assert G1.number_of_nodes() > 0, "G1 should have at least one node"
    assert G2.number_of_nodes() > 0, "G2 should have at least one node"
    assert G1.number_of_edges() > 0, "G1 should have at least one edge"
    assert G2.number_of_edges() > 0, "G2 should have at least one edge"
    assert G1.number_of_nodes() == G2.number_of_nodes(), "G1 and G2 should have the same number of nodes"

    return hungarian_algorithm(
        make_GRAMPA_similarity_matrix_vectorized(
            nx.adjacency_matrix(G1).toarray(), nx.adjacency_matrix(G2).toarray(), bandwidth))


def make_degree_alignment_permutation(G1: nx.graph, G2:nx.graph) -> np.ndarray:
    """
    Makes a permutation vector that aligns the degree sequences of two graphs G1 and G2. 
    !VIBECODED!

    Parameters:

    G1 (nx.Graph): First graph
    G2 (nx.Graph): Second graph

    Returns:

    np.array: Permutation vector that aligns the degree sequences of G1 and G2.
    """

    assert isinstance(G1, nx.Graph), "G1 should be a networkx graph"
    assert isinstance(G2, nx.Graph), "G2 should be a networkx graph"
    assert G1.number_of_nodes() > 0, "G1 should have at least one node"
    assert G2.number_of_nodes() > 0, "G2 should have at least one node"
    assert G1.number_of_edges() > 0, "G1 should have at least one edge"
    assert G2.number_of_edges() > 0, "G2 should have at least one edge"
    assert G1.number_of_nodes() == G2.number_of_nodes(), "G1 and G2 should have the same number of nodes"

    degrees_G1 = np.array([d for _, d in G1.degree()])
    degrees_G2 = np.array([d for _, d in G2.degree()])

    perm1 = np.argsort(degrees_G1)
    perm2 = np.argsort(degrees_G2)

    return compose_permutations(perm2, inverse_permutation(perm1))


from GRASP.functionalMaps import functional_maps


def make_GRASP_permutation(G1: nx.Graph, G2: nx.Graph,
                q: int = 20, k: int = 20,
                lower_t: float = 0.1, upper_t: float = 50.0,
                linsteps: bool = True,
                laa: int = 3,
                icp: bool = False,
                icp_its: int = 3) -> list[int]:
    """
    Align two networkx graphs using GRASP. https://github.com/AU-DIS/GRASP

    Returns
    -------
    perm : list[int]  — perm[i] = node in G2 matched to node i in G1
    """
    A1 = nx.to_numpy_array(G1)
    A2 = nx.to_numpy_array(G2)

    matching, _, _, _ = functional_maps(
        A1, A2,
        q=q, k=k,
        laa=laa,          # 1=nearest neighbour, 2=sort-greedy, 3=hungarian (best)
        icp=icp,
        icp_its=icp_its,
        lower_t=lower_t,
        upper_t=upper_t,
        linsteps=linsteps,
        graph_name='',    # unused inside functional_maps, just a label
        i=0, j=0          # unused loop indices
    )

    # matching is a dict {node_in_G1: node_in_G2}; convert to ordered list
    n = A1.shape[0]
    return inverse_permutation(np.array([matching[i] for i in range(n)]))


def laplacian_eigenvectors(G):
    """Returns eigenvectors of the graph Laplacian sorted by eigenvalue (non-decreasing)."""
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(L)  # eigh guarantees real, sorted eigenvalues
    return eigenvectors  # columns are eigenvectors, sorted by eigenvalue


def make_umeyama_permutation(G1, G2):
    """
    Approximates the orthogonal matrix O1 @ O2.T with a permutation matrix
    using Umeyama's fix: |O1| @ |O2|.T, then solving the LAP.

    Parameters
    ----------
    G1, G2 : networkx graphs with the same number of nodes N

    Returns
    -------
    perm : np.ndarray of shape (N,)
        perm[i] = j means node i in G1 corresponds to node j in G2.
    """
    O1 = laplacian_eigenvectors(G1)  # shape (N, N)
    O2 = laplacian_eigenvectors(G2)  # shape (N, N)

    # Umeyama similarity: entry-wise absolute values
    X = np.abs(O1) @ np.abs(O2).T  # shape (N, N)

    # Solve LAP: maximize trace(P.T @ X) <=> minimize trace(P.T @ (-X))
    row_ind, col_ind = sp.optimize.linear_sum_assignment(-X)

    # row_ind is always [0, 1, ..., N-1]; col_ind is the permutation
    return col_ind









