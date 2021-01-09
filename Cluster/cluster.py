#library
import pandas as pd
import numpy as np
import sys

from tqdm import tqdm # appear the precess of running situation.
import time

from scipy.spatial.distance import pdist, squareform

#0. Data Load
data = pd.read_csv(sys.argv[1], delimiter='\t') # Load train (input text file)

#1. Data Preprocessing
all_elements = [index for index in data.index] # Save index name.

#Make a distance metrix to compute dissimilarity.
distance_matrix = pdist(data, metric='euclidean')
dissimilarity_matrix = np.array(squareform(distance_matrix))
#dissimilarity_matrix = pd.DataFrame(squareform(distance_matrix), columns=all_elements, index=all_elements)
print(dissimilarity_matrix)

#2. Modeling : DIANA Clustering
#2-1. Compute dissimilarity average in ONE Cluster. 
def avg_dissim_within_group_element(node, element_list):
    max_diameter = -np.inf
    sum_dissm = 0 #Set Sum equal zero.
    for i in element_list: 
        sum_dissm += dissimilarity_matrix[node][i] #While iterate element_list, Sum the distance matrix value singly in a node.
        if( dissimilarity_matrix[node][i]  > max_diameter): #If distance matrix is bigger than max_distance,
            max_diameter = dissimilarity_matrix[node][i] # that distance matrix value become a max_diameter.
    if(len(element_list)>1):
        avg = sum_dissm/(len(element_list)-1) # Average of distance matrix.
    else: 
        avg = 0
    return avg

# 2-2. Compute dissimilarity average between different Group(e.g. Cluster1 and Cluster2) 
# id in sperated new group  = splinter_list
def avg_dissim_across_group_element(node, main_list, splinter_list):
    if len(splinter_list) == 0: #there is no spliter group, return zero.
        return 0  
    sum_dissm = 0
    for j in splinter_list:
        sum_dissm = sum_dissm + dissimilarity_matrix[node][j] #Compute average between Object in splinter group 
    avg = sum_dissm/(len(splinter_list))  #and all object dissimilarity matrix.
    return avg

# 2-3. Cluster Splinter
def splinter(main_list, splinter_group):
    most_dissm_object_value = -np.inf #initate minus.
    most_dissm_object_index = None
    for node in main_list:
        x = avg_dissim_within_group_element(node, main_list) # Previously, a point in main group as a standard.
        y = avg_dissim_across_group_element(node, main_list, splinter_group) # a point in the seperated group.
        diff = x - y # difference between X and Y
        if diff > most_dissm_object_value:
            most_dissm_object_value = diff
            most_dissm_object_index = node # save index and value which has largest value between two groups.
    if(most_dissm_object_value>0): # differnce is Plus, Create new splinter group. flag = 1
        return  (most_dissm_object_index, 1)
    else: # difference is minus, flag = -1
        return (-1, -1)

# 2-4. Split
def split(element_list):
    main_list = element_list
    splinter_group = []    
    (most_dissm_object_index, flag) = splinter(main_list, splinter_group)
    while(flag > 0): # Iterate splinter function until a flag become minus.
        main_list.remove(most_dissm_object_index) #Delete the most largest dissimilarity average object index in the main list.
        splinter_group.append(most_dissm_object_index) # Then, append in the new splinter group.
        (most_dissm_object_index, flag) = splinter(element_list, splinter_group)
    
    return (main_list, splinter_group)

# 2-5. look for maximum distance in the current cluster.
def max_distance(cluster_list):
    max_diameter_cluster_index = None
    max_diameter_cluster_value = -np.inf
    index = 0
    for element_list in cluster_list:
        for i in element_list: #columns
            for j in element_list: #rows
                #Switch the largest dissimilarity average object(index), value. 
                if dissimilarity_matrix[i][j]  > max_diameter_cluster_value: 
                    max_diameter_cluster_value = dissimilarity_matrix[i][j]
                    max_diameter_cluster_index = index
        
        index +=1
    
    if(max_diameter_cluster_value <= 0):
        return -1
    
    return max_diameter_cluster_index

# main
if __name__ == '__main__':

    # Save arguments list
    argv = sys.argv 

    # Set the number of cluster.
    num_clusters = sys.argv[-1]
    current_clusters = ([all_elements])
    print(current_clusters)
    level = 1
    index = 0

    with tqdm(total=100) as pbar:
        while((index!=-1) and (level!=num_clusters)): #Proceed until the index equal -1 and setting number of cluster.
            (a_clstr, b_clstr) = split(current_clusters[index])
            del current_clusters[index] # Delete current cluster.
            current_clusters.append(a_clstr) #original cluster
            current_clusters.append(b_clstr) #splinter cluster
            index = max_distance(current_clusters)
            level +=1
            pbar.update(10)

    for i in range(num_clusters): # Save the results.
        pd.DataFrame(current_clusters[i], columns=['id']).to_csv("%s_cluster_%d.txt" %(sys.argv[1], i), sep='\t')   
