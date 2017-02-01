from .utils import Atom, Residue, ActiveSite
import numpy as np
from scipy import linalg
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


def get_tail_coords(site):
    """
    For a given active site, go through the residues
    and return a list of coordinates for the last atoms
    of each residue. The last atom is simply the last
    element in the list of atoms for each residue in the
    active site

    Input: ActiveSite
    Output: List of Coordinates
    """
    residues = site.residues
    coords = []
    for residue in residues:
        tail_atom = residue.atoms[-1]
        coords.append(tail_atom.coords) 
    return coords

def find_dist(site):
    """
    For a given active site, find the average distance
    from the best fit plane through the coordinates of the
    tail atoms as defined above (in function get_tail_coords).

    Input: ActiveSite
    Output: average distance (float)
    """
    
    # Get the coordinates of the tail 
    # atoms as defined above (get_tail_coords)
    coords = get_tail_coords(site)
    coords = np.asarray(coords)

    # If only three tail atoms, distance will
    # be defined as 0.0
    if len(coords) <= 3:
        return 0.0
    else:
        # Define the 3D least-squares problem. We are solving the least
        # squares coefficients C where AC = y where y are the coordinates
        # of the tail atoms
        A = np.c_[coords[:,0], coords[:,1], np.ones(coords.shape[0])]
        C,sum_residuals,_,_ = linalg.lstsq(A, coords[:,2])

        # Define the average distance as the sum of the squared
        # residuals normalized by the number of tail atoms
        avg_dist = sum_residuals / len(coords)

        return avg_dist


def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.
    Note that here, a larger similarity means the two sites are
    ostensibly more different. This is accounted for when
    the similarity matrix is computed

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """

    # Get the average distance for both active sites
    dist_a = find_dist(site_a)
    dist_b = find_dist(site_b)
    # Define Similarity as the absolute value
    # of the difference between the average
    # distances of the two active sites
    similarity = abs(dist_a - dist_b)

    # Set to logarithmic scale since the 
    # range of distances might be very large
    if similarity <= 1.0:
        similarity = 0.0
    else:
        similarity = np.log(similarity)

    return similarity

def get_sim_matrix(active_sites):
    """
    For a given active site, return the similarity
    matrix, sim_matrix, where sim_matrix[i,j] is 
    the similarity score between active site i and
    active site j

    Input: List of ActiveSites
    Output: Similarity Matrix (dictionary of diciontaries)
    """

    # Initiate matrix of zeros
    # sim_matrix = np.zeros((len(active_sites), len(active_sites)))
    sim_matrix = {}

    # Loop through the list of active sites and calculate the similarity
    # scores, filling out the similarity matrix. Since the matrix is
    # symmetric, we have that sim_matrix[i,j] = sim_matrix[j,i]

    # max_score keeps track of the biggest score found
    # for normalizing
    max_score = 0.0
    for i in range(len(active_sites)):
        for j in range(i, len(active_sites)):
            site_a = active_sites[i]
            site_b = active_sites[j]
            similarity = compute_similarity(site_a, site_b)
            if similarity > max_score:
                max_score = similarity

            if site_a in sim_matrix:
                sim_matrix[site_a][site_b] = similarity
            else:
                sim_matrix[site_a] = {site_b : similarity}

            if site_b in sim_matrix:
                sim_matrix[site_b][site_a] = similarity
            else:
                sim_matrix[site_b] = {site_a : similarity}

    # loop through the sim_matrix once to normalize all 
    # the values and set to [0,1] scale where 1 is the best
    # possible similarity score
    for outer_site in sim_matrix:
        for inner_site in sim_matrix[outer_site]:
            current_sim = sim_matrix[outer_site][inner_site]
            sim_matrix[outer_site][inner_site] = 1.0 - (current_sim / max_score)

    return sim_matrix

def compute_center_similarity(active_site, cluster, sim_matrix):
    """
    Computes the similarity of the active_site with the 
    cluster center as the average similarity between
    the active_site and all the nodes in the cluster

    Inputs: 
        ActiveSite
        Cluster (list of ActiveSites)

    Output:
        Distance from active_site to cluster center (float)
    """

    sim = 0.0
    for site in cluster:
        sim += sim_matrix[active_site][site]
    sim = sim / float(len(cluster))
    return sim

def compute_cluster_centers(clusters, sim_matrix):
    cluster_centers = []

    for cluster in clusters:
        best_center = cluster[0]
        best_sim = compute_center_similarity(cluster[0], cluster, sim_matrix)
        for active_site in cluster:
            sim = compute_center_similarity(active_site, cluster, sim_matrix)
            if sim > best_sim:
                best_sim = sim
                best_center = active_site
        cluster_centers.append(best_center)

    return cluster_centers

def k_means(k, active_sites, sim_matrix):
    # Make a copy of the active_site list
    active_sites_working = list(active_sites)
    
    # random.shuffle(active_sites_working)

    if k > len(active_sites):
        raise ValueError('K-means Clustering - Number of Cluster Centers'+ 
                         'cannot be larger than number of active sites')

    # Create initial Clusters
    clusters = []
    clusters_one_pass = []
    cluster_centers = []
    for i in range(k):
        cluster_centers.append(active_sites_working.pop())
        clusters.append([cluster_centers[i]])

    # Iteratively assign active sites to clusters until we reach the
    # point at which the resulting cluster configuration
    # has been seen either one or two passes previously
    while(True):

        new_clusters = []
        for i in range(k):
            new_clusters.append([])
        active_sites_working = list(active_sites)
        for i, center in enumerate(cluster_centers):
            active_sites_working.remove(center)
            new_clusters[i].append(center)
        for active_site in active_sites_working:
            # best_sim_score = compute_center_similarity(active_site, clusters[0], sim_matrix)
            best_sim_score = sim_matrix[active_site][cluster_centers[0]]
            best_cluster = 0
            for i in range(1, k):
                cluster = clusters[i]
                # print(i, clusters)
                # center_similarity = compute_center_similarity(active_site, cluster, sim_matrix)
                center_similarity = sim_matrix[active_site][cluster_centers[i]]
                # print(center_similarity)
                if center_similarity > best_sim_score:
                    best_sim_score = center_similarity
                    best_cluster = i
            new_clusters[best_cluster].append(active_site)



        if new_clusters == clusters or new_clusters == clusters_one_pass:
            break
        else:
            cluster_centers = compute_cluster_centers(new_clusters, sim_matrix)
            clusters_one_pass = clusters
            clusters = new_clusters

    return clusters

def get_optimal_k(active_sites, sim_matrix):
    """
    For use in determining a good k value to
    use for cluster_by_partitioning. Does k-means
    clustering for a range of k values and plots
    the inter-cluster similarity

    Input: list of ActiveSites
    Output: Plot with Inter cluster similarites
    """

    clustering_sims = []
    # k_vals = range(1,100)
    top_limit =  math.ceil(len(active_sites) * (3/5))
    k_vals = range(1, top_limit)
    # k_vals = [8]
    # print(k_vals)
    for k in k_vals:
        avg_cluster_sim = 0.0
        # clusterings.append(cluster_by_partitioning(active_sites, k))
        clustering = k_means(k, active_sites, sim_matrix)
        for cluster in clustering:
            cluster_sim = 0.0
            for active_site in cluster: 
                cluster_sim += compute_center_similarity(active_site, cluster, sim_matrix)
            cluster_sim = cluster_sim / float(len(cluster))
            avg_cluster_sim += cluster_sim

        avg_cluster_sim = avg_cluster_sim / float(k)
        clustering_sims.append(avg_cluster_sim)



    max_sim = np.amax(clustering_sims)
    best_k = 1
    for k in k_vals:
        if clustering_sims[k - 1] >= max_sim * 0.90:
            best_k = k
            break
    # fig = plt.figure()
    # plt.plot(k_vals, clustering_sims)
    # plt.plot([best_k], [clustering_sims[best_k - 1]], 'r*')
    # plt.xlabel('k')
    # plt.ylabel('Average Inter-Cluster Similarity')
    # plt.show()
    return best_k


def cluster_by_partitioning(active_sites):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances, k (int)
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    
    # Get similarity matrix for the list of active sites. 
    sim_matrix = get_sim_matrix(active_sites)

    # For determining good k value to use
    best_k = get_optimal_k(active_sites, sim_matrix)

    # See pdf for details. After multiple runs determined 20
    # to be a good value for k
    # k = 20
    clusters = k_means(best_k, active_sites, sim_matrix)
    return clusters


def cluster_hierarchically(active_sites):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    # Fill in your code here!

    return []
