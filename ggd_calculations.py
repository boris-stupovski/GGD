import GGD
import time
import pickle

def calculate_ggd_GRAMPA_same(graphs, sizes):
    out = {}
    for key1 in graphs.keys():
        for key2 in graphs.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]) and ((key1[1]==key2[1] and key1[2]<key2[2]) or (key1[1]<key2[1])):
                print(f"Processing {key1, key2}...")
                G1 = graphs[key1]
                G2 = graphs[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_with_relabeling(G1, G2, GGD.make_GRAMPA_permutation(G1, G2, 0.2))
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_ggd_natural_same(graphs, sizes):
    out = {}
    for key1 in graphs.keys():
        for key2 in graphs.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]) and ((key1[1]==key2[1] and key1[2]<key2[2]) or (key1[1]<key2[1])):
                print(f"Processing {key1, key2}...")
                G1 = graphs[key1]
                G2 = graphs[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized(G1, G2)
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_ggd_GRASP_same(graphs, sizes):
    out = {}
    for key1 in graphs.keys():
        for key2 in graphs.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]) and ((key1[1]==key2[1] and key1[2]<key2[2]) or (key1[1]<key2[1])):
                print(f"Processing {key1, key2}...")
                G1 = graphs[key1]
                G2 = graphs[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_with_relabeling(G1,G2,GGD.make_GRASP_permutation(G1,G2))
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_ggd_random_same(graphs, sizes):
    out = {}
    for key1 in graphs.keys():
        for key2 in graphs.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]) and ((key1[1]==key2[1] and key1[2]<key2[2]) or (key1[1]<key2[1])):
                print(f"Processing {key1, key2}...")
                G1 = graphs[key1]
                G2 = graphs[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_with_relabeling(G1, G2, GGD.make_random_permutation(key1[0]))
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_ggd_degree_same(graphs, sizes):
    out = {}
    for key1 in graphs.keys():
        for key2 in graphs.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]) and ((key1[1]==key2[1] and key1[2]<key2[2]) or (key1[1]<key2[1])):
                print(f"Processing {key1, key2}...")
                G1 = graphs[key1]
                G2 = graphs[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_with_relabeling(G1, G2, GGD.make_degree_alignment_permutation(G1, G2))
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_ggd_umeyama_same(graphs, sizes):
    out = {}
    for key1 in graphs.keys():
        for key2 in graphs.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]) and ((key1[1]==key2[1] and key1[2]<key2[2]) or (key1[1]<key2[1])):
                print(f"Processing {key1, key2}...")
                G1 = graphs[key1]
                G2 = graphs[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_with_relabeling(G1, G2, GGD.make_umeyama_permutation(G1,G2))
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_ggd_GRAMPA_different(graphs1, graphs2, sizes):
    out = {}
    for key1 in graphs1.keys():
        for key2 in graphs2.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]):
                print(f"Processing {key1, key2}...")
                G1 = graphs1[key1]
                G2 = graphs2[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_with_relabeling(G1, G2, GGD.make_GRAMPA_permutation(G1, G2, 0.2))
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_ggd_random_different(graphs1, graphs2, sizes):
    out = {}
    for key1 in graphs1.keys():
        for key2 in graphs2.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]):
                print(f"Processing {key1, key2}...")
                G1 = graphs1[key1]
                G2 = graphs2[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_with_relabeling(G1, G2, GGD.make_random_permutation(key1[0]))
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_ggd_degree_different(graphs1, graphs2, sizes):
    out = {}
    for key1 in graphs1.keys():
        for key2 in graphs2.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]):
                print(f"Processing {key1, key2}...")
                G1 = graphs1[key1]
                G2 = graphs2[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_with_relabeling(G1, G2, GGD.make_degree_alignment_permutation(G1, G2))
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_ggd_GRASP_different(graphs1, graphs2, sizes):
    out = {}
    for key1 in graphs1.keys():
        for key2 in graphs2.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]):
                print(f"Processing {key1, key2}...")
                G1 = graphs1[key1]
                G2 = graphs2[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_with_relabeling(G1,G2,GGD.make_GRASP_permutation(G1,G2))
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_ggd_umeyama_different(graphs1, graphs2, sizes):
    out = {}
    for key1 in graphs1.keys():
        for key2 in graphs2.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]):
                print(f"Processing {key1, key2}...")
                G1 = graphs1[key1]
                G2 = graphs2[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_with_relabeling(G1, G2, GGD.make_umeyama_permutation(G1,G2))
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_bounds_same(graphs, sizes):
    out = {}
    for key1 in graphs.keys():
        for key2 in graphs.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]) and ((key1[1]==key2[1] and key1[2]<key2[2]) or (key1[1]<key2[1])):
                print(f"Processing {key1, key2}...")
                G1 = graphs[key1]
                G2 = graphs[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_bounds(G1, G2)
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out


def calculate_bounds_different(graphs1, graphs2, sizes):
    out = {}
    for key1 in graphs1.keys():
        for key2 in graphs2.keys():
            if (key1[0] in sizes) and (key1[0]==key2[0]):
                print(f"Processing {key1, key2}...")
                G1 = graphs1[key1]
                G2 = graphs2[key2]
                start_time = time.time()
                out[(key1[0], key1[1], key2[1], key1[2], key2[2])] = GGD.GGD_normalized_bounds(G1, G2)
                end_time = time.time()
                print(f"Done in {end_time - start_time:.2f} seconds.")
    return out





#timing start
start_time = time.time()

#reading graphs
with open('data/graphs/ER_graphs.pkl', 'rb') as f:
    ER_graphs = pickle.load(f)

with open('data/graphs/BA_graphs.pkl', 'rb') as f:
    BA_graphs = pickle.load(f)

sizes = [512, 1024, 2048]

#calculating bounds 
ER_bounds = calculate_bounds_same(ER_graphs, sizes)
BA_bounds = calculate_bounds_same(BA_graphs, sizes)
ER_BA_bounds = calculate_bounds_different(ER_graphs, BA_graphs, sizes)

#calculating GGD for GRAMPA aligned 
ER_grampa = calculate_ggd_GRAMPA_same(ER_graphs, sizes)
BA_grampa = calculate_ggd_GRAMPA_same(BA_graphs, sizes)
ER_BA_grampa = calculate_ggd_GRAMPA_different(ER_graphs, BA_graphs, sizes)

#calculating GGD for random permutations
ER_random = calculate_ggd_random_same(ER_graphs, sizes)
BA_random = calculate_ggd_random_same(BA_graphs, sizes)
ER_BA_random = calculate_ggd_random_different(ER_graphs, BA_graphs, sizes)

#calculating GGD for degree aligned same graphs
ER_degree = calculate_ggd_degree_same(ER_graphs, sizes)
BA_degree = calculate_ggd_degree_same(BA_graphs, sizes)
ER_BA_degree = calculate_ggd_degree_different(ER_graphs, BA_graphs, sizes)

#calculating GGD for Umeyama aligned same graphs
ER_umeyama = calculate_ggd_umeyama_same(ER_graphs, sizes)
BA_umeyama = calculate_ggd_umeyama_same(BA_graphs, sizes)
ER_BA_umeyama = calculate_ggd_umeyama_different(ER_graphs, BA_graphs, sizes)

#calculating GGD for GRASP aligned same graphs
ER_grasp = calculate_ggd_GRASP_same(ER_graphs, sizes)
BA_grasp = calculate_ggd_GRASP_same(BA_graphs, sizes)
ER_BA_grasp = calculate_ggd_GRASP_different(ER_graphs, BA_graphs, sizes)

#calculating GGD for natural alignment of BA graphs
BA_natural = calculate_ggd_natural_same(BA_graphs, sizes)

#saving results

#bounds
with open('data/ggd_calculation_results/ER_bounds.pkl', 'wb') as f:
    pickle.dump(ER_bounds, f)
with open('data/ggd_calculation_results/BA_bounds.pkl', 'wb') as f:
    pickle.dump(BA_bounds, f)
with open('data/ggd_calculation_results/ER_BA_bounds.pkl', 'wb') as f:
    pickle.dump(ER_BA_bounds, f)

#GRAMPA
with open('data/ggd_calculation_results/ER_GRAMPA.pkl', 'wb') as f:
    pickle.dump(ER_grampa, f)
with open('data/ggd_calculation_results/BA_GRAMPA.pkl', 'wb') as f:
    pickle.dump(BA_grampa, f)
with open('data/ggd_calculation_results/ER_BA_GRAMPA.pkl', 'wb') as f:
    pickle.dump(ER_BA_grampa, f)

#GRASP
with open('data/ggd_calculation_results/ER_GRASP.pkl', 'wb') as f:
    pickle.dump(ER_grasp, f)
with open('data/ggd_calculation_results/BA_GRASP.pkl', 'wb') as f:
    pickle.dump(BA_grasp, f)
with open('data/ggd_calculation_results/ER_BA_GRASP.pkl', 'wb') as f:
    pickle.dump(ER_BA_grasp, f)

#umeyama
with open('data/ggd_calculation_results/ER_UMEYAMA.pkl', 'wb') as f:
    pickle.dump(ER_umeyama, f)
with open('data/ggd_calculation_results/BA_UMEYAMA.pkl', 'wb') as f:
    pickle.dump(BA_umeyama, f)
with open('data/ggd_calculation_results/ER_BA_UMEYAMA.pkl', 'wb') as f:
    pickle.dump(ER_BA_umeyama, f)

#random
with open('data/ggd_calculation_results/ER_random.pkl', 'wb') as f:
    pickle.dump(ER_random, f)
with open('data/ggd_calculation_results/BA_random.pkl', 'wb') as f:
    pickle.dump(BA_random, f)
with open('data/ggd_calculation_results/ER_BA_random.pkl', 'wb') as f:
    pickle.dump(ER_BA_random, f) 

#degree
with open('data/ggd_calculation_results/ER_degree.pkl', 'wb') as f:
    pickle.dump(ER_degree, f)
with open('data/ggd_calculation_results/BA_degree.pkl', 'wb') as f:
    pickle.dump(BA_degree, f)
with open('data/ggd_calculation_results/ER_BA_degree.pkl', 'wb') as f:
    pickle.dump(ER_BA_degree, f)

#natural
with open('data/ggd_calculation_results/BA_natural.pkl', 'wb') as f:
    pickle.dump(BA_natural, f)

#timing end
end_time = time.time()

print(f"Total time: {(end_time - start_time)/60.:.2f} minutes.")