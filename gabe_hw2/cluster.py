from .utils import Atom, Residue, ActiveSite
import numpy as np
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy.signal import argrelextrema


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

    # Grab the residues and upack the coordinates
    residues = site.residues
    coords = []

    for residue in residues:
        # Tail atoms are defined as the 
        # last element in the list of atoms 
        # for the given residue
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
    the similarity matrix is computed.

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
    Output: Similarity Matrix (dictionary of dictionaries)
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

def get_avg_sim(site, cluster, sim_matrix):
    """
    Compute the average similarity between the ActiveSite
    site and the cluster

    Input: site (ActiveSite)
    Output: cluster (list of ActiveSites)
    """

    sim = 0.0
    for cluster_site in cluster:
        sim += sim_matrix[site][cluster_site]

    sim = sim / float(len(cluster))
    return sim

def compute_cluster_centers(clusters, sim_matrix):
    """
    Given a clustering, return the cluster centers
    defined as the elements in each cluster that are most
    similar to all other elements in that same cluster.

    Input: Clustering (list of lists of ActiveSite instances)
           sim_matrix (a dictionary of dictionaries with similarity scores)
    
    Output: cluster_centers (A list of ActiveSite instances)
    """
    cluster_centers = []

    # For each cluster, find the cluster member that has the highest
    # average similarity to all other nodes in the same cluster. 
    # Define this as the center of the respective cluster
    for cluster in clusters:
        best_center = cluster[0]
        best_sim = get_avg_sim(cluster[0], cluster, sim_matrix)
        for active_site in cluster:
            sim = get_avg_sim(active_site, cluster, sim_matrix)
            if sim > best_sim:
                best_sim = sim
                best_center = active_site
        cluster_centers.append(best_center)

    # Return list of ActiveSites that preserves the
    # ordering of the input clusters. cluster_centers[i]
    # is the ActiveSite which is found to be the center of
    # the cluster clusters[i]
    return cluster_centers

def k_means(k, active_sites, sim_matrix):
    """
    Perform k-means clustering given a list of ActiveSites, 
    a similarity matrix, and a given value of k

    Input: k (int)
           active_sites (list of ActiveSites)
           sim_matrix (dictionary of dictionaries)

    Output: clusters (a list of lists of ActiveSites)
    """

    # Make a copy of the active_site list
    active_sites_working = list(active_sites)
    

    # Make sure k is at most the number of ActiveSites
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

        # The working copy of the next clustering
        new_clusters = []

        # Iniitalize new_clusters with empty clusters
        for i in range(k):
            new_clusters.append([])
        active_sites_working = list(active_sites)

        # Add cluster centers to their respective clusters
        for i, center in enumerate(cluster_centers):
            active_sites_working.remove(center)
            new_clusters[i].append(center)

        # Assign every site that has not already been designated
        # as a cluster center to cluster with the most similar 
        # center
        for active_site in active_sites_working:
            # best_sim_score = get_avg_sim(active_site, clusters[0], sim_matrix)
            best_sim_score = sim_matrix[active_site][cluster_centers[0]]
            best_cluster = 0
            for i in range(1, k):
                cluster = clusters[i]
                center_similarity = sim_matrix[active_site][cluster_centers[i]]
                if center_similarity > best_sim_score:
                    best_sim_score = center_similarity
                    best_cluster = i
            new_clusters[best_cluster].append(active_site)


        # If this clustering has been seen in the last two iterations,
        # exit the loop
        if new_clusters == clusters or new_clusters == clusters_one_pass:
            break
        else:
            cluster_centers = compute_cluster_centers(new_clusters, sim_matrix)
            clusters_one_pass = clusters
            clusters = new_clusters

    return clusters

def get_optimal_k(active_sites, sim_matrix, plot = False):
    """
    For use in determining a good k value to
    use for cluster_by_partitioning. Does k-means
    clustering for a range of k values and chooses
    one

    Input: active_sites (list of ActiveSites)
           sim_matrix (dictionary of dictionaries)
    Output: k (int)
    """

    clustering_sims = []
    top_limit =  int(math.ceil(len(active_sites) * (3.0/5)))
    k_vals = range(1, top_limit)

    # Iterate over a range of k values and find the average cluster similarity,
    # defined as the average similarity between nodes within a cluster averaged 
    # across all clusters for every resulting clustering
    for k in k_vals:
        avg_cluster_sim = 0.0
        # clusterings.append(cluster_by_partitioning(active_sites, k))
        clustering = k_means(k, active_sites, sim_matrix)
        for cluster in clustering:
            cluster_sim = 0.0
            for active_site in cluster: 
                cluster_sim += get_avg_sim(active_site, cluster, sim_matrix)
            cluster_sim = cluster_sim / float(len(cluster))
            avg_cluster_sim += cluster_sim

        avg_cluster_sim = avg_cluster_sim / float(k)
        clustering_sims.append(avg_cluster_sim)


    # If we allow k to get high enoug, the
    # intra-cluster similarity will converge to
    # 1 since every ActiveSite will be assigned to
    # its own cluster. So, find the lowest k value
    # at which the resulting intra-cluster similarity 
    # is at 90% of the max intra-cluster similarity found
    max_sim = np.amax(clustering_sims)
    best_k = 1
    for k in k_vals:
        if clustering_sims[k - 1] >= max_sim * 0.90:
            best_k = k
            break


    # Plotting k values and resulting intra-cluster similarities
    if plot == True:
        fig = plt.figure()
        plt.plot(k_vals, clustering_sims)
        plt.plot([best_k], [clustering_sims[best_k - 1]], 'r*')
        plt.xlabel('k')
        plt.ylabel('Average Inter-Cluster Similarity')
        plt.title('Tested k values for Clustering by Partitioning')
        plt.show()


    return best_k

def compute_worst_similarity(cluster_1, cluster_2, sim_matrix):
    """
    Given two clusters, computes the worst similarity score between 
    any possible pair of nodes between the two clusters

    Input: cluster1, cluster2 (lists of of ActiveSites)
           sim_matrix (dictionary of dictionaries)
    Output: worst_sim (a float between 0.0 and 1.0)
    """
    worst_sim = 1.0
    for site_1 in cluster_1:
        for site_2 in cluster_2:
            current_sim = sim_matrix[site_1][site_2]
            if current_sim < worst_sim:
                worst_sim = current_sim

    return worst_sim


def furthest_neighbor(active_sites, sim_matrix, similarity_cutoff):
    """
    Perform furthest_neighbor hierarchical agglomerative clustering 
    given a similarity cutoff value. Begins by placing every ActiveSite
    into its own cluster then iteratively joining the two clusters between
    which the least similar elements have the highest similarity score among
    all respective clusters in the current clustering. Iteratively does this 
    until the best similarity score between furthest cluster neighbors is
    less than the provided similarity cutoff value

    Input: active_sites (list of ActiveSites)
           sim_matrix (dictionary of dictionaries)
           similarity_cutoff (float)

    Output: clusters (a list of lists of ActiveSites)
    """
    # clusterings = []


    # Initialize current_clustering such that
    # every ActiveSite is its own cluster
    current_clustering = []
    for site in active_sites:
        current_clustering.append([site])

    # clusterings.append(current_clustering)

    while True:
        best_sim = 0.0
        join_index_1 = 0
        join_index_2 = 1

        # Go through every cluster and find the furthest-neighbor
        # similarity scores for every other cluster for all pairs
        # that haven't been checked yet. If a furhtest-neighbor
        # similarity score is found that is better than the 
        # current best score, update the current best score
        # and keep track of which two clusters produce this
        # new best score
        for i in range(len(current_clustering)):
            current_cluster = current_clustering[i]
            for j in range(i + 1, len(current_clustering)):
                comparison_cluster = current_clustering[j]
                worst_sim = compute_worst_similarity(current_cluster, comparison_cluster, sim_matrix)
                if worst_sim > best_sim:
                    best_sim = worst_sim
                    join_index_1 = i
                    join_index_2 = j

        # If, this current iteration's best furthest-neighbor similarity score
        # is worse than the provided similarity cutoff, exit the loop
        if best_sim <= similarity_cutoff: 
            break

        # Initialize a new clustering
        new_clustering = []
        # Create new cluster by joining the two clusters in the previous clustering that
        # produced the best furthest-neighbor similarity score
        joined_cluster = current_clustering[join_index_1] + current_clustering[join_index_2]
        # Add all old clusters from the previous clustering that were not joined
        # to the new clustering
        for cluster_index, current_cluster in enumerate(current_clustering):
            if (cluster_index != join_index_1 and cluster_index != join_index_2):
                new_clustering.append(current_cluster)
        # Add the new joined cluster
        new_clustering.append(joined_cluster)
        # clusterings.append(current_clustering)
        current_clustering = new_clustering

    return current_clustering

def get_sim_cutoff(active_sites, sim_matrix, plot = False):
    """
    For use in determining a good similarity cutoff value
    to use for cluster_hierarchically. Does hierarchical
    clustering over a range of possible similarity cutoff
    values and chooses one

    Input: active_sites (list of ActiveSites)
           sim_matrix (dictionary of dictionaries)
    Output: best_sim_cutoff (float)
    """
    sim_cutoffs = np.linspace(0.0,1, 30)
    clustering_sims = []

    # Iterate through range of possible sim_cutoff values and
    # perform hierarchical clustering for each one. For each 
    # resulting clustering, calculate the average intra-cluster 
    # similarity score averaged across all clusters
    for current_sim_cutoff in sim_cutoffs:
        avg_cluster_sim = 0.0
        # clusterings.append(cluster_by_partitioning(active_sites, k))
        # clustering = k_means(k, active_sites, sim_matrix)
        clustering = furthest_neighbor(active_sites, sim_matrix, current_sim_cutoff)
        for cluster in clustering:
            cluster_sim = 0.0
            for active_site in cluster: 
                cluster_sim += get_avg_sim(active_site, cluster, sim_matrix)
            cluster_sim = cluster_sim / float(len(cluster))
            avg_cluster_sim += cluster_sim

        avg_cluster_sim = avg_cluster_sim / len(clustering)
        clustering_sims.append(avg_cluster_sim)


    
    best_sim_cutoff = 0.0
    best_cluster_sim = 0.0
    best_avg_diff = 0.0
    clustering_sims_array = np.asarray(clustering_sims)

    # For every clustering (and respective similarity cutoff value), 
    # look for local maxima in intra-cluster similarities (for which
    # the intra-cluster similarity is higher than both the next highest
    # and next lowest similarity cutoff scores). Find similarity cutoff
    # value that produces the local intra-cluster similarity maximum whose
    # average distance from the neighboring intra-cluster similarities is
    # higher than for all other local maxima
    for index, clustering_sim in enumerate(clustering_sims):
        if index == 0 or index == (len(clustering_sims) - 1):
            continue

        prev_point = clustering_sims[index - 1]
        next_point = clustering_sims[index + 1]
        if (clustering_sim > prev_point) and (clustering_sim > next_point):
            avg_diff = ((clustering_sim - prev_point) + (clustering_sim - next_point)) / 2.0
            if avg_diff > best_avg_diff:
                best_avg_diff = avg_diff
                best_sim_cutoff = sim_cutoffs[index]
                best_cluster_sim = clustering_sim


    # Plotting for visualizing similarity cutoff inputs and
    # resulting intra-cluster similarities
    if plot == True:
        fig = plt.figure()
        plt.plot(sim_cutoffs, clustering_sims)
        plt.plot(best_sim_cutoff, best_cluster_sim, 'r*')
        plt.xlabel('similarity cutoff')
        plt.ylabel('Average Inter-Cluster Similarity')
        plt.title('Tested Similarity Cutoffs for Hierarchical clustering')
        plt.show()

    return best_sim_cutoff


def cluster_by_partitioning(active_sites, sim_matrix):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances, k (int)
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    
    if len(active_sites) <= 1:
        clusters = []
        for site in active_sites:
            clusters.append([site])
        return clusters
    # For determining good k value to use
    best_k = get_optimal_k(active_sites, sim_matrix)
    # Use k-means implementation above to get clusters from partitioning
    clusters = k_means(best_k, active_sites, sim_matrix)
    return clusters


def cluster_hierarchically(active_sites, sim_matrix):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    if len(active_sites) <= 1:
        clusters = []
        for site in active_sites:
            clusters.append([site])
        return clusters

    # Determine a good similarity cutoff score to use as defined above
    similarity_cutoff = get_sim_cutoff(active_sites, sim_matrix)
    # Use furthest neighbor agglomerative clustering to get clusters from 
    # hierarchical clustering
    clusterings = furthest_neighbor(active_sites, sim_matrix, similarity_cutoff)
    return clusterings

def quality_score(clustering, active_sites, sim_matrix):
    """
    Measure the quality of the clustering by taking the average silhouette
    score of all ActiveSites in the clustering. For a given ActiveSite, the 
    silhouette score s is defined as
            s = (a - b) / (min(a, b))
    where a is the average similarity of the ActiveSite with all other members
    of its assinged cluster and b is the average similarity between the ActiveSite
    and all others members of the closest cluster (defined as having the highest
    average similarity)

    Input: clustering (list of lists of ActiveSites)
    Output: Silhouette Score (float bewteen -1 and 1)
    """

    s = 0.0

    for site in active_sites:
        # Find the cluster to which the current site belongs
        for cluster in clustering:
            if site in cluster:
                break

        # Compute the a value
        a = get_avg_sim(site, cluster, sim_matrix)

        # Compute the b value
        b = 0.0
        for test_cluster in clustering:
            if test_cluster != cluster:
                test_sim = get_avg_sim(site, test_cluster, sim_matrix)
                if test_sim > b:
                    b = test_sim

        s += (a - b) / min(a, b)

    s = s / len(active_sites)
    return s


def comparison_score(hierarchical_clustering, partioning_clustering, active_sites, sim_matrix):
    """
    Given two clusterings, one hiearchical and one partioning, find their similarity defined as determined
    by the difference between their two respective silouette scores and normalized to the range [0,1] where
    1 means the two clusterings have the same silouette scores and 0 means that the two clusterings have
    the largest possible difference in silouette scores

    Input: hierarchical_clustering, partioning_clustering (list of lists of ActiveSites)
           active_sites (list of ActiveSites)
           sim_matrix (dictionary of dictionaries)

    Output: comparison_score (float)
    """

    # Get the two respective silhouette scores
    h_quality = quality_score(hierarchical_clustering, active_sites, sim_matrix)
    p_quality = quality_score(partioning_clustering, active_sites, sim_matrix)

    # Find their difference and normalize to [0,1]
    comparison_score = 1.0 - abs(h_quality - p_quality) / 2.0
    return comparison_score

def compare(hierarchical_clustering, partioning_clustering, active_sites, sim_matrix):
    """
    Given a hierarchical and partioning clustering, produce a plot that measures their
    respesctive qualities

    Intput: hierarchical_clustering, partioning_clustering (list of lists of ActiveSites)
            active_sites (list of ActiveSites)
            sim_matrix (dictionary of dictionaries)

    Output: None
    """

    partitioning_qualities = []
    partioning_clusterings = []
    top_limit =  int(math.ceil(len(active_sites) * (3.0/5)))
    k_vals = range(1, top_limit)

    hierarchical_qualities = []
    hierarchical_clusterings = []
    sim_cutoffs = np.linspace(0.0,1, len(k_vals))

    comparison_scores = []

    # Calculate partioning and hierarchical clusterings across range of k values and
    # similarity cutoffs. For each resulting clustering, calculate the quality scores
    for k in k_vals:
        partioning_clustering = k_means(k, active_sites, sim_matrix)
        partioning_clusterings.append(partioning_clustering)
        quality = quality_score(partioning_clustering, active_sites, sim_matrix)
        partitioning_qualities.append(quality)
    for current_sim_cutoff in sim_cutoffs:
        hierarchical_clustering = furthest_neighbor(active_sites, sim_matrix, current_sim_cutoff)
        hierarchical_clusterings.append(hierarchical_clustering)
        quality = quality_score(hierarchical_clustering, active_sites, sim_matrix)
        hierarchical_qualities.append(quality)

    # Calculate the comparison scores for clusterings produced at the same position in the
    # range [0,1] of input values (k value for partioning or similary cutoff score for hierarchical)
    for i in range(len(k_vals)):
        score = comparison_score(hierarchical_clusterings[i], partioning_clusterings[i], active_sites, sim_matrix)
        comparison_scores.append(score)

    # Plot the resulting scores
    k_vals_normalized = np.asarray(k_vals) / np.max(k_vals)
    fig = plt.figure()
    plt.plot(k_vals_normalized, partitioning_qualities, label = 'Partitioning Qualities')
    plt.plot(sim_cutoffs, hierarchical_qualities, label = 'Hierarchical Qualities')
    plt.plot(k_vals_normalized, comparison_scores, label = 'Comparison Scores')
    plt.xlabel('normalized k (Partitioning) and similarity cutoffs (Hierarchical)')
    plt.ylabel('Quality and Comparison Scores')
    plt.title('Quality and Comparison Scores for Partioning and Hierarchical Clustering')
    plt.legend()
    plt.show()


def biological_score(cluster):
    """
    Produces a measure of how biologically significant a given cluster in the 
    following manner. For every pair of active sites in the cluster, compute 
    a sequence similarity score between the two by dividing the number of residues
    the two sequences have in common by the length of the longer sequence. This 
    produces a number in the range [0, 1] for every pair of active sites. Average
    this score across the entire cluster and return as the biological score of the 
    cluster.

    Input: cluster (list of ActiveSites)

    Output: biological_score (float)
    """
    
    biological_score = 0.0
    # For every pair of active sites in the 
    # cluster, find the residues (types of residues) in their
    # sequences and count the number of residues that they
    # have in common. 
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            pair_score = 0.0
            residues_1 = []
            residues_2 = []
            for residue in cluster[i].residues:
                residues_1.append(residue.type)
            for residue in cluster[j].residues:
                residues_2.append(residue.type)

            for residue_type in set(residues_1):
                pair_score += residues_2.count(residue_type)

            # Divide the number of residues they have in common
            # by the length of the longer sequence
            pair_score = pair_score / max(len(residues_1), len(residues_2))

            biological_score += pair_score
            
    if len(cluster) > 1:
        biological_score = biological_score / (0.5 * (len(cluster) - 1) * len(cluster))


    return biological_score


def clustering_biology(clustering, active_sites, sim_matrix):
    """
    Given a clustering, calculate the average cluster biological
    significance score across all clusters

    Intput: clustering (list of lists of ActiveSites)
            active_sites (list of ActiveSites)
            sim_matrix (dictionary of dictionaries)

    Output: avg_bio_score (float)
    """

    # Calculate biological score for every cluster in 
    # clustering then average
    avg_bio_score = 0.0
    for cluster in clustering:
        avg_bio_score += biological_score(cluster)

    avg_bio_score = avg_bio_score / float(len(clustering))
    return avg_bio_score

def compare_biology_significance(hierarchical_clustering, partioning_clustering, active_sites, sim_matrix):
    """
    Given a hierarchical and partioning clustering, produce a plot that measures their
    respesctive qualities

    Intput: hierarchical_clustering, partioning_clustering (list of lists of ActiveSites)
            active_sites (list of ActiveSites)
            sim_matrix (dictionary of dictionaries)

    Output: None
    """

    partitioning_biology_scores = []
    partioning_clusterings = []
    top_limit =  int(math.ceil(len(active_sites) * (3.0/5)))
    k_vals = range(1, top_limit)

    hierarchical_biology_scores = []
    hierarchical_clusterings = []
    sim_cutoffs = np.linspace(0.0,1, len(k_vals))


    # Calculate partioning and hierarchical clusterings across range of k values and
    # similarity cutoffs. For each resulting clustering, calculate the biological 
    # significance scores
    for k in k_vals:
        partioning_clustering = k_means(k, active_sites, sim_matrix)
        partioning_clusterings.append(partioning_clustering)
        biology = clustering_biology(partioning_clustering, active_sites, sim_matrix)
        partitioning_biology_scores.append(biology)
    for current_sim_cutoff in sim_cutoffs:
        hierarchical_clustering = furthest_neighbor(active_sites, sim_matrix, current_sim_cutoff)
        hierarchical_clusterings.append(hierarchical_clustering)
        biology = clustering_biology(hierarchical_clustering, active_sites, sim_matrix)
        hierarchical_biology_scores.append(biology)


    # Plot the resulting biology scores for all resulting clusterings
    k_vals_normalized = np.asarray(k_vals) / np.max(k_vals)
    fig = plt.figure()
    plt.plot(k_vals_normalized, partitioning_biology_scores, label = 'Partitioning')
    plt.plot(sim_cutoffs, hierarchical_biology_scores, label = 'Hierarchical')
    plt.xlabel('normalized k (Partitioning) and similarity cutoffs (Hierarchical)')
    plt.ylabel('Biological Significance Scores')
    plt.title('Biological Significance for Partioning and Hierarchical Clustering')
    plt.legend()
    plt.show()
    
