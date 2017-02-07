from gabe_hw2 import cluster
from gabe_hw2 import io
import os

def test_similarity():
    filename_a = os.path.join("data", "276.pdb")
    filename_b = os.path.join("data", "4629.pdb")

    activesite_a = io.read_active_site(filename_a)
    activesite_b = io.read_active_site(filename_b)

    assert cluster.compute_similarity(activesite_a, activesite_a) == 0.0
    assert cluster.compute_similarity(activesite_a, activesite_b) == cluster.compute_similarity(activesite_b, activesite_a)


def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    sim_matrix = cluster.get_sim_matrix(active_sites)

    assert cluster.cluster_by_partitioning([], {}) == []
    assert cluster.cluster_by_partitioning([active_sites[0]], {}) == [[active_sites[0]]]
    clustering = cluster.cluster_by_partitioning(active_sites, sim_matrix)
    assert len(clustering) == 1
    s = set(active_sites)
    assert set(clustering[0]) == s

def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    sim_matrix = cluster.get_sim_matrix(active_sites)

    # update this assertion
    assert cluster.cluster_hierarchically([], {}) == []
    assert cluster.cluster_hierarchically([active_sites[0]], {}) == [[active_sites[0]]]
    clustering = cluster.cluster_hierarchically(active_sites, sim_matrix)
    assert len(clustering) == 2
    assert clustering[0] == [active_sites[2]]
    assert set(clustering[1]) == set([active_sites[0], active_sites[1]])