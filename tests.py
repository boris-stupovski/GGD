import GGD
import pytest
import numpy as np
import scipy as sp
import networkx as nx


def generate_random_SPD_matrix(size:int)-> np.ndarray:
    """
    Generates a random Symmetric Positive Definite (SPD) matrix of given size.
    
    Parameters:
    size (int): The size of the matrix to generate.
    
    Returns:
    np.ndarray: A random SPD matrix of shape (size, size).
    """
    assert isinstance(size, int), "Size should be an integer"
    assert size > 0, "Size should be a positive integer"
    
    A = np.random.normal(size=(size, size))

    return np.dot(A, A.T) + np.eye(size)  


testdata_zero = [
    (generate_random_SPD_matrix(10)),
    (generate_random_SPD_matrix(11)),
    (generate_random_SPD_matrix(100)),
    (generate_random_SPD_matrix(101)),
    (generate_random_SPD_matrix(500)),
    (generate_random_SPD_matrix(501)),
]
@pytest.mark.parametrize("A", testdata_zero)
def test_AIRM_zero(A):
    assert np.isclose(GGD.AIRM(A,A),0.)


testdata_symmetric = [
    (generate_random_SPD_matrix(10),generate_random_SPD_matrix(10)),
    (generate_random_SPD_matrix(11),generate_random_SPD_matrix(11)),
    (generate_random_SPD_matrix(100),generate_random_SPD_matrix(100)),
    (generate_random_SPD_matrix(101),generate_random_SPD_matrix(101)),
    (generate_random_SPD_matrix(500),generate_random_SPD_matrix(500)),
    (generate_random_SPD_matrix(501),generate_random_SPD_matrix(501)),
]
@pytest.mark.parametrize("A,B", testdata_symmetric)
def test_AIRM_symmetric(A,B):
    assert np.isclose(GGD.AIRM(A,B),GGD.AIRM(B,A))


testdata_triangle = [
    (generate_random_SPD_matrix(10), generate_random_SPD_matrix(10), generate_random_SPD_matrix(10)),
    (generate_random_SPD_matrix(11), generate_random_SPD_matrix(11), generate_random_SPD_matrix(11)),
    (generate_random_SPD_matrix(100), generate_random_SPD_matrix(100), generate_random_SPD_matrix(100)),
    (generate_random_SPD_matrix(101), generate_random_SPD_matrix(101), generate_random_SPD_matrix(101)),
    (generate_random_SPD_matrix(500), generate_random_SPD_matrix(500), generate_random_SPD_matrix(500)),
    (generate_random_SPD_matrix(501), generate_random_SPD_matrix(501), generate_random_SPD_matrix(501)),
]
@pytest.mark.parametrize("A,B,C", testdata_triangle)
def test_AIRM_triangle(A,B,C):
    assert GGD.AIRM(A,B) <= GGD.AIRM(A,C) + GGD.AIRM(B,C)


def AIRM_with_identity(A:np.array)->float:
    return np.sqrt(np.sum(np.power(np.log(sp.linalg.eigh(A,eigvals_only=True)),2)))


testdata_identity = [
    (generate_random_SPD_matrix(10)),
    (generate_random_SPD_matrix(11)),
    (generate_random_SPD_matrix(100)),
    (generate_random_SPD_matrix(101)),
    (generate_random_SPD_matrix(500)),
    (generate_random_SPD_matrix(501)),
]
@pytest.mark.parametrize("A", testdata_identity)
def test_AIRM_identity(A):
    assert np.isclose(GGD.AIRM(A,np.eye(A.shape[0])),AIRM_with_identity(A))


A = np.array([[1, 0, 0, 0], 
              [0, 2, 0, 0],
              [0, 0, 3, 0],
              [0, 0, 0, 4]])

B = np.array([[4, 0, 0, 0], 
              [0, 3, 0, 0],
              [0, 0, 2, 0],
              [0, 0, 0, 1]])

C = np.array([[3, -1, 0, -1], 
              [-1, 3, -1, 0],
              [0, -1, 3, -1],
              [-1, 0, -1, 3]])

D = np.array([[4, -1, -1, -1], 
              [-1, 2, 0, 0],
              [-1, 0, 2, 0],
              [-1, 0, 0, 2]])

testdata_examples = [(A,B,2.04265),(A,C,1.624),(A,D,1.85974),
                    (B,C,1.624),(B,D,1.3212),(C,D,0.980192)]
@pytest.mark.parametrize("A,B,expected", testdata_examples)
def test_AIRM_example(A,B,expected):
    """
    Test the AIRM function with a specific examples.
    """
    assert np.isclose(GGD.AIRM(A,B), expected)
  

def create_KN_diagonal(N:int)->np.array:
    K = np.zeros((N,N))
    np.fill_diagonal(K, N)
    K[0,0] = 0
    return K


testdata_eigenbasis_matrix = [
    (10),
    (11),
    (51),
    (52),
    (101),
    (102),
]
@pytest.mark.parametrize("N", testdata_eigenbasis_matrix)
def test_basis_generation(N):
    G = nx.complete_graph(N)
    Q = nx.laplacian_matrix(G).toarray()
    S = GGD.get_K_eigenbasis_matrix(N)
    assert np.isclose(np.dot(S.T, np.dot(Q, S)), create_KN_diagonal(N)).all()
    assert np.isclose(np.dot(S.T, S), np.eye(N)).all()


testdata_get_SPD_laplacian_submatrix_KN = [
    (10),
    (11),
    (51),
    (52),
    (101),
    (102),
]
@pytest.mark.parametrize("N", testdata_eigenbasis_matrix)
def test_get_SPD_laplacian_submatrix_KN(N):
    G = nx.complete_graph(N)
    Q = GGD.get_SPD_laplacian_submatrix(GGD.get_connected_laplacian(G))
    assert np.allclose(Q, N*np.eye(N-1))


testdata_get_SPD_laplacian_submatrix_BA = [
    (nx.barabasi_albert_graph(10, 1)),
    (nx.barabasi_albert_graph(11, 1)),
    (nx.barabasi_albert_graph(10, 2)),
    (nx.barabasi_albert_graph(11, 2)),
    (nx.barabasi_albert_graph(100, 1)),
    (nx.barabasi_albert_graph(101, 1)),
    (nx.barabasi_albert_graph(100, 1)),
    (nx.barabasi_albert_graph(101, 2)),
    (nx.barabasi_albert_graph(10, 3)),
    (nx.barabasi_albert_graph(11, 3)),
    (nx.barabasi_albert_graph(10, 4)),
    (nx.barabasi_albert_graph(11, 4)),    
    (nx.barabasi_albert_graph(10, 1)),
    (nx.barabasi_albert_graph(100, 3)),
    (nx.barabasi_albert_graph(100, 4)),
    (nx.barabasi_albert_graph(101, 3)),
    (nx.barabasi_albert_graph(101, 4)),
    (nx.barabasi_albert_graph(200, 1)),
    (nx.barabasi_albert_graph(200, 2)),
    (nx.barabasi_albert_graph(200, 3)),
    (nx.barabasi_albert_graph(200, 4)),
    (nx.barabasi_albert_graph(200, 5)),
    (nx.barabasi_albert_graph(200, 6)),
    (nx.barabasi_albert_graph(200, 7)),
]
@pytest.mark.parametrize("G", testdata_get_SPD_laplacian_submatrix_BA)
def test_get_SPD_laplacian_submatrix_BA(G):
    assert np.allclose(sp.linalg.eigh(GGD.get_SPD_laplacian_submatrix(GGD.get_connected_laplacian(G)), eigvals_only=True), 
                       sp.linalg.eigh(GGD.get_connected_laplacian(G), eigvals_only=True)[1:])
    

testdata_get_GGD_K_BA = [
    (nx.barabasi_albert_graph(10, 1)),
    (nx.barabasi_albert_graph(11, 1)),
    (nx.barabasi_albert_graph(10, 2)),
    (nx.barabasi_albert_graph(11, 2)),
    (nx.barabasi_albert_graph(100, 1)),
    (nx.barabasi_albert_graph(101, 1)),
    (nx.barabasi_albert_graph(100, 1)),
    (nx.barabasi_albert_graph(101, 2)),
    (nx.barabasi_albert_graph(10, 3)),
    (nx.barabasi_albert_graph(11, 3)),
    (nx.barabasi_albert_graph(10, 4)),
    (nx.barabasi_albert_graph(11, 4)),    
    (nx.barabasi_albert_graph(10, 1)),
    (nx.barabasi_albert_graph(100, 3)),
    (nx.barabasi_albert_graph(100, 4)),
    (nx.barabasi_albert_graph(101, 3)),
    (nx.barabasi_albert_graph(101, 4)),
    (nx.barabasi_albert_graph(200, 1)),
    (nx.barabasi_albert_graph(200, 2)),
    (nx.barabasi_albert_graph(200, 3)),
    (nx.barabasi_albert_graph(200, 4)),
    (nx.barabasi_albert_graph(200, 5)),
    (nx.barabasi_albert_graph(200, 6)),
    (nx.barabasi_albert_graph(200, 7)),
]
@pytest.mark.parametrize("G", testdata_get_GGD_K_BA)
def test_GGD_K_BA(G):
    N = G.number_of_nodes()
    K = nx.complete_graph(N)
    eigvals = sp.linalg.eigh(GGD.get_connected_laplacian(G), eigvals_only=True)[1:]
    assert np.allclose(GGD.GGD(G,K), np.sqrt(np.sum(np.power(np.log(eigvals/N),2))))


testdata_GGD_K_P = [
    (10),
    (11),
    (51),
    (52),
    (101),
    (102),
    (1001),
    (1002),
]
@pytest.mark.parametrize("N", testdata_GGD_K_P)
def test_GGD_K_P(N):
    P = nx.path_graph(N)
    K = nx.complete_graph(N)
    eigvals = 2* (1 - np.cos(np.pi / N * np.arange(1, N)))
    assert np.allclose(GGD.GGD(P,K), np.sqrt(np.sum(np.power(np.log(eigvals/N),2))))


testdata_permutations_orthogonal = [
    (10),
    (11),
    (51),
    (52),
    (101),
    (102),
    (1001),
    (1002),
]


testdata_AIRM_permutations_invariance = [
    (generate_random_SPD_matrix(10),generate_random_SPD_matrix(10),GGD.make_random_permutation(10)),
    (generate_random_SPD_matrix(11),generate_random_SPD_matrix(11),GGD.make_random_permutation(11)), 
    (generate_random_SPD_matrix(100),generate_random_SPD_matrix(100),GGD.make_random_permutation(100)),
    (generate_random_SPD_matrix(101),generate_random_SPD_matrix(101),GGD.make_random_permutation(101)),
    (generate_random_SPD_matrix(500),generate_random_SPD_matrix(500),GGD.make_random_permutation(500)),
    (generate_random_SPD_matrix(501),generate_random_SPD_matrix(501),GGD.make_random_permutation(501)),
]
@pytest.mark.parametrize("A,B,P", testdata_AIRM_permutations_invariance)
def testd_AIRM_permutations_invariance(A,B,P):
    assert np.isclose(GGD.AIRM(A,B),
                      GGD.AIRM(GGD.similarity_transform_permutation(A,P),GGD.similarity_transform_permutation(B,P)))


testdata_AIRM_permutations_covariance = [
    (generate_random_SPD_matrix(10),generate_random_SPD_matrix(10),GGD.make_random_permutation(10)),
    (generate_random_SPD_matrix(11),generate_random_SPD_matrix(11),GGD.make_random_permutation(11)), 
    (generate_random_SPD_matrix(100),generate_random_SPD_matrix(100),GGD.make_random_permutation(100)),
    (generate_random_SPD_matrix(101),generate_random_SPD_matrix(101),GGD.make_random_permutation(101)),
    (generate_random_SPD_matrix(500),generate_random_SPD_matrix(500),GGD.make_random_permutation(500)),
    (generate_random_SPD_matrix(501),generate_random_SPD_matrix(501),GGD.make_random_permutation(501)),
]
@pytest.mark.parametrize("A,B,P", testdata_AIRM_permutations_invariance)
def testd_AIRM_permutations_covariance(A,B,P):
    assert np.isclose(GGD.AIRM(GGD.similarity_transform_permutation(A,P),B),
                      GGD.AIRM(A,GGD.similarity_transform_permutation(B,GGD.inverse_permutation(P))))
    

testdata_bounds_star=[
    (10),
    (11),
    (51),
    (52),
    (101),
    (102),
    (1001),
    (1002),
]
@pytest.mark.parametrize("N", testdata_bounds_star)
def test_GGD_bounds_star(N):
    S = nx.star_graph(N-1)
    assert np.isclose(GGD.GGD_bounds(S,S)[0],0)
    assert np.isclose(GGD.GGD_bounds(S,S)[1], np.sqrt(2)*np.log(N))


testdata_bounds_path=[
    (10),
    (11),
    (51),
    (52),
    (101),
    (102),
    (1001),
    (1002),
]
@pytest.mark.parametrize("N", testdata_bounds_path)
def test_GGD_bounds_path(N):
    P = nx.path_graph(N)
    eigvals = 2* (1 - np.cos(np.pi / N * np.arange(1, N)))
    assert np.isclose(GGD.GGD_bounds(P,P)[0],0)
    assert np.isclose(GGD.GGD_bounds(P,P)[1], np.sqrt(np.sum(np.power(np.log(eigvals/eigvals[::-1]),2))))


testdata_bounds_cycle_WS2_graph=[
    (10),
    (11),
    (51),
    (52),
    (101),
    (102),
    (1001),
    (1002),
]
@pytest.mark.parametrize("N", testdata_bounds_cycle_WS2_graph)
def test_GGD_bounds_cycle_WS(N):
    C = nx.cycle_graph(N)
    WS2 = nx.watts_strogatz_graph(N, 4, 0)
    eigvals_C = np.sort(2* (1 - np.cos(2 * np.pi / N * np.arange(1, N))))
    eigvals_WS2 = np.sort(5 - np.sin(5 * np.pi / N * np.arange(1, N))/np.sin(np.pi / N * np.arange(1, N)))
    assert np.isclose(GGD.GGD_bounds(C,WS2)[0], np.sqrt(np.sum(np.power(np.log(eigvals_C/eigvals_WS2),2))))


#test GGD path complete graph
sizes = [10, 11, 50, 51, 100, 101]
@pytest.mark.parametrize("size", sizes)
def test_GGD_path_complete(size):
    G1 = nx.path_graph(size)
    G2 = nx.complete_graph(size)
    eigvals_path = 2* (1 - np.cos(np.pi / size * np.arange(1, size)))
    eigvals_complete = np.array([size]* (size-1))
    assert np.isclose(round(GGD.GGD(G1,G2),10), round(np.sqrt(np.sum(np.power(np.log(eigvals_path/eigvals_complete),2))),10))


#test GGD star complete graph
sizes = [10, 11, 50, 51, 100, 101]
@pytest.mark.parametrize("size", sizes)
def test_GGD_star_complete(size):
    G1 = nx.star_graph(size-1)
    G2 = nx.complete_graph(size)
    eigvals_star = np.array([1]* (size-2) + [size])
    eigvals_complete = np.array([size]* (size-1))
    assert np.isclose(round(GGD.GGD(G1,G2),10), round(np.sqrt(np.sum(np.power(np.log(eigvals_star/eigvals_complete),2))),10))


#test GGD with relabeling
sizes = [10, 11, 50, 51, 100, 101]
@pytest.mark.parametrize("size", sizes)
def test_GGD_relabeling_stars(size):
    for size in sizes:
        G1 = nx.star_graph(size-1)
        G2 = nx.star_graph(size-1)
        perm = np.arange(size)
        perm[0], perm[1] = perm[1], perm[0]
        assert np.isclose(round(GGD.GGD_with_relabeling(G1,G2,perm),3), round(np.sqrt(2)*np.log(size),3))


#test GGD_with_relabeling path complete graphs
permutations = [ GGD.make_random_permutation(100), 
                 GGD.make_random_permutation(100), 
                 GGD.make_random_permutation(100), 
                 GGD.make_random_permutation(100),
                 GGD.make_random_permutation(100)]
@pytest.mark.parametrize("perm", permutations)
def test_GGD_with_relabeling_path_complete(perm):
    size = 100
    G1 = nx.path_graph(size)
    G2 = nx.complete_graph(size)
    eigvals_path = 2* (1 - np.cos(np.pi / size * np.arange(1, size)))
    eigvals_complete = np.array([size]* (size-1))
    assert np.isclose(round(GGD.GGD_with_relabeling(G1,G2,perm),10), round(np.sqrt(np.sum(np.power(np.log(eigvals_path/eigvals_complete),2))),10))


#test GGD_with_relabeling complete path
permutations = [ GGD.make_random_permutation(100), 
                 GGD.make_random_permutation(100), 
                 GGD.make_random_permutation(100), 
                 GGD.make_random_permutation(100),
                 GGD.make_random_permutation(100)]
@pytest.mark.parametrize("perm", permutations)
def test_GGD_with_relabeling_complete_path(perm):
    size = 100
    G2 = nx.path_graph(size)
    G1 = nx.complete_graph(size)
    eigvals_path = 2* (1 - np.cos(np.pi / size * np.arange(1, size)))
    eigvals_complete = np.array([size]* (size-1))
    assert np.isclose(round(GGD.GGD_with_relabeling(G1,G2,perm),10), round(np.sqrt(np.sum(np.power(np.log(eigvals_path/eigvals_complete),2))),10))



# Testing make_GRAMPA_similarity_matrix function
def test_similarity_matrix_diagonal_1111_1122_bandwith10():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 2]])
    
    Ex = np.array([[1, 1, 1/2, 1/2], 
                   [1, 1, 1/2, 1/2],
                   [1, 1, 1/2, 1/2],
                   [1, 1, 1/2, 1/2]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix(A, B, 1.), Ex)


def test_similarity_matrix_diagonal_1111_1122_bandwith05():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 2]])
    
    Ex = np.array([[4, 4, 4/5, 4/5], 
                   [4, 4, 4/5, 4/5],
                   [4, 4, 4/5, 4/5],
                   [4, 4, 4/5, 4/5]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix(A, B, 0.5), Ex)
                            

def test_similarity_matrix_diagonal_1111_1234_bandwith10():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
    
    Ex = np.array([[1, 1/2, 1/5, 1/10], 
                   [1, 1/2, 1/5, 1/10],
                   [1, 1/2, 1/5, 1/10],
                   [1, 1/2, 1/5, 1/10]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix(A, B, 1.0), Ex)


def test_similarity_matrix_diagonal_1111_1234_bandwith05():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
    
    Ex = np.array([[4, 4/5, 4/17, 4/37], 
                   [4, 4/5, 4/17, 4/37],
                   [4, 4/5, 4/17, 4/37],
                   [4, 4/5, 4/17, 4/37]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix(A, B, 0.5), Ex)


def test_similarity_matrix_diagonal_1122_1234_bandwith10():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 2]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
    
    Ex = np.array([[1, 1/2, 1/5, 1/10], 
                   [1, 1/2, 1/5, 1/10],
                   [1/2, 1, 1/2, 1/5],
                   [1/2, 1, 1/2, 1/5]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix(A, B, 1.0), Ex)


def test_similarity_matrix_diagonal_1122_1234_bandwith05():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 2]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
    
    Ex = np.array([[4, 4/5, 4/17, 4/37], 
                   [4, 4/5, 4/17, 4/37],
                   [4/5, 4, 4/5, 4/17],
                   [4/5, 4, 4/5, 4/17]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix(A, B, 0.5), Ex)


# Testing make_GRAMPA_similarity_matrix function VECTORIZED
def test_similarity_matrix_diagonal_1111_1122_bandwith10_VECTORIZED():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 2]])
    
    Ex = np.array([[1, 1, 1/2, 1/2], 
                   [1, 1, 1/2, 1/2],
                   [1, 1, 1/2, 1/2],
                   [1, 1, 1/2, 1/2]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix_vectorized(A, B, 1.), Ex)


def test_similarity_matrix_diagonal_1111_1122_bandwith05_VECTORIZED():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 2]])
    
    Ex = np.array([[4, 4, 4/5, 4/5], 
                   [4, 4, 4/5, 4/5],
                   [4, 4, 4/5, 4/5],
                   [4, 4, 4/5, 4/5]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix_vectorized(A, B, 0.5), Ex)
                            

def test_similarity_matrix_diagonal_1111_1234_bandwith10_VECTORIZED():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
    
    Ex = np.array([[1, 1/2, 1/5, 1/10], 
                   [1, 1/2, 1/5, 1/10],
                   [1, 1/2, 1/5, 1/10],
                   [1, 1/2, 1/5, 1/10]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix_vectorized(A, B, 1.0), Ex)


def test_similarity_matrix_diagonal_1111_1234_bandwith05_VECTORIZED():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
    
    Ex = np.array([[4, 4/5, 4/17, 4/37], 
                   [4, 4/5, 4/17, 4/37],
                   [4, 4/5, 4/17, 4/37],
                   [4, 4/5, 4/17, 4/37]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix_vectorized(A, B, 0.5), Ex)


def test_similarity_matrix_diagonal_1122_1234_bandwith10_VECTORIZED():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 2]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
    
    Ex = np.array([[1, 1/2, 1/5, 1/10], 
                   [1, 1/2, 1/5, 1/10],
                   [1/2, 1, 1/2, 1/5],
                   [1/2, 1, 1/2, 1/5]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix_vectorized(A, B, 1.0), Ex)


def test_similarity_matrix_diagonal_1122_1234_bandwith05_VECTORIZED():
    A = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 2]])

    B = np.array([[1, 0, 0, 0], 
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
    
    Ex = np.array([[4, 4/5, 4/17, 4/37], 
                   [4, 4/5, 4/17, 4/37],
                   [4/5, 4, 4/5, 4/17],
                   [4/5, 4, 4/5, 4/17]])
    
    assert np.allclose(GGD.make_GRAMPA_similarity_matrix_vectorized(A, B, 0.5), Ex)


# Testing cs_entrywise_scalar_product function
def cs_entrywise_scalar_product(A:np.ndarray, B:np.ndarray) -> float:
    """
    Calculates the cosinse similarity between two matrices A and B using entrywise scalar product.
    c(A,B)=<A,B>/sqrt(<A,A>*<B,B>)

    Parameters:
    
    A (np.array): First matrix
    B (np.array): Second matrix

    Returns:
    
    float: Cosine similarity between A and B

    """
    assert isinstance(A, np.ndarray), "A should be a numpy array"
    assert isinstance(B, np.ndarray), "B should be a numpy array"
    assert len(A.shape) == 2, "A should be a 2D matrix"
    assert len(B.shape) == 2, "B should be a 2D matrix"
    assert A.shape == B.shape, "A and B should have the same shape"
    assert A.shape[0] == A.shape[1], "A should be a square matrix"
    assert B.shape[0] == B.shape[1], "B should be a square matrix"
    assert not np.allclose(A, 0), "A should not be a zero matrix"
    assert not np.allclose(B, 0), "B should not be a zero matrix"

    return np.sum(A * B) / np.sqrt(np.sum(A * A) * np.sum(B * B))

testdata_same = [
    (generate_random_SPD_matrix(10)),
    (generate_random_SPD_matrix(11)),
    (generate_random_SPD_matrix(100)),
    (generate_random_SPD_matrix(101)),
    (generate_random_SPD_matrix(500)),
    (generate_random_SPD_matrix(501)),
]
@pytest.mark.parametrize("A", testdata_zero)
def test_cs_entrywise_scalar_product_same_matrix(A):
    assert np.isclose(cs_entrywise_scalar_product(A,A), 1.)


testdata_I_J = [
    (np.eye(10),np.ones((10,10)), np.power(10, -0.5)),
    (np.eye(11),np.ones((11,11)), np.power(11, -0.5)),
    (np.eye(100),np.ones((100,100)), np.power(100, -0.5)),
    (np.eye(101),np.ones((101,101)), np.power(101, -0.5)),
    (np.eye(500),np.ones((500,500)), np.power(500, -0.5)),
    (np.eye(501),np.ones((501,501)), np.power(501, -0.5)),
]
@pytest.mark.parametrize("A,B,expected", testdata_I_J)
def test_cs_entrywise_scalar_product_I_J(A,B,expected):
    assert np.isclose(cs_entrywise_scalar_product(A,B), expected)


def test_cs_small_examples():
    A = np.array([[1, 0], 
                  [0, 1]])
    B = np.array([[2, 0], 
                  [0, 2]])
    assert np.isclose(cs_entrywise_scalar_product(A,B), 1.)

    C = np.array([[1, 0], 
                  [0, 2]])
    D = np.array([[2, 0], 
                  [0, 1]])
    assert np.isclose(cs_entrywise_scalar_product(C,D), 0.8)


sizes = [4,5,10,11,50,51,100,101]
@pytest.mark.parametrize("size", sizes)
def test_cs_path_star(size):
    G1 = nx.path_graph(size)
    G2 = nx.star_graph(size-1)
    A1 = nx.adjacency_matrix(G1).toarray()
    A2 = nx.adjacency_matrix(G2).toarray()
    assert np.isclose(cs_entrywise_scalar_product(A1,A2), 1/(size-1))


#test hungarian algorithm random permutations
def make_permutation_matrix(p:np.ndarray) -> np.ndarray:
    return np.eye(len(p))[p,:]   


def make_random_permutation_matrix(N:int) -> np.ndarray:
    return make_permutation_matrix(np.random.permutation(N))

permutation_matrices = [make_random_permutation_matrix(4), 
                        make_random_permutation_matrix(5), 
                        make_random_permutation_matrix(10),
                        make_random_permutation_matrix(11),
                        make_random_permutation_matrix(50),
                        make_random_permutation_matrix(51),
                        make_random_permutation_matrix(100),
                        make_random_permutation_matrix(101)
                        ]
@pytest.mark.parametrize("P", permutation_matrices)
def test_hungarian_algorithm_PMs(P):
    P_recovered = make_permutation_matrix(GGD.hungarian_algorithm(P))
    assert np.allclose(P, P_recovered)

#test hungarian algoriht Kuhn example
def test_hungrian_algorithm_Kuhn_example():
    X = np.array([[8, 7, 9, 9],
                  [5, 2, 7, 8],
                  [6, 1, 4, 9],
                  [2, 3, 2, 6]])
    P = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 1, 0, 0]])
    P_recovered = make_permutation_matrix(GGD.hungarian_algorithm(X))
    assert np.allclose(P, P_recovered)


#dense ER GRAMPA tests 
sizes = [100, 101, 200, 201]
@pytest.mark.parametrize("size", sizes)
def test_dense_ER_GRAMPA_A(size):
    p = 0.9
    G1 = nx.erdos_renyi_graph(size, p)
    A1 = nx.adjacency_matrix(G1).toarray()
    A2 = GGD.similarity_transform_permutation(nx.adjacency_matrix(G1).toarray(),GGD.make_random_permutation(size))
    G2 = nx.from_numpy_array(A2)
    A3 = GGD.similarity_transform_permutation(nx.adjacency_matrix(G2).toarray(),GGD.make_GRAMPA_permutation(G1,G2,0.2))
    assert cs_entrywise_scalar_product(A1,A3) > 0.99


#testing degree alignement
#star 5
A = np.array([[0, 1, 1, 1, 1],
              [1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0]])
B = GGD.similarity_transform_permutation(A, GGD.make_random_permutation(5))
GA = nx.from_numpy_array(A)
GB = nx.from_numpy_array(B)
def test_degree_alignment_star5():
    assert np.isclose(0., GGD.GGD_normalized_with_relabeling(GA, GB, GGD.make_degree_alignment_permutation(GA, GB)))

#star 100
S = nx.star_graph(99)
A = nx.adjacency_matrix(S).toarray()
B = GGD.similarity_transform_permutation(A, GGD.make_random_permutation(100))
GA = nx.from_numpy_array(A)
GB = nx.from_numpy_array(B)
def test_degree_alignment_star100():
    assert np.isclose(0., GGD.GGD_normalized_with_relabeling(GA, GB, GGD.make_degree_alignment_permutation(GA, GB)))

#adhoc test for degree alignment
A = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
              [1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
              [1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
              [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0 ,0]])
B = GGD.similarity_transform_permutation(A, GGD.make_random_permutation(10))
GA = nx.from_numpy_array(A)
GB = nx.from_numpy_array(B)
def test_degree_alignment_adhoc():
    assert np.isclose(0., GGD.GGD_normalized_with_relabeling(GA, GB, GGD.make_degree_alignment_permutation(GA, GB)))


#dense ER umeyama tests 
sizes = [100, 101, 200, 201]
@pytest.mark.parametrize("size", sizes)
def test_dense_ER_umeyama_A(size):
    p = 0.9
    G1 = nx.erdos_renyi_graph(size, p)
    A1 = nx.adjacency_matrix(G1).toarray()
    A2 = GGD.similarity_transform_permutation(nx.adjacency_matrix(G1).toarray(),GGD.make_random_permutation(size))
    G2 = nx.from_numpy_array(A2)
    A3 = GGD.similarity_transform_permutation(nx.adjacency_matrix(G2).toarray(),GGD.make_umeyama_permutation(G1,G2))
    assert cs_entrywise_scalar_product(A1,A3) > 0.99