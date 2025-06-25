# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:18:21 2023

@author: tcava
"""


import numpy as np

def input_eos(dict_list):

    # Get all unique keys from the dictionaries
    unique_keys = set(key for d in dict_list for key in d.keys())
    
    # Create the array of floats with dimensions (number of unique keys, n)
    num_unique_keys = len(unique_keys)
    n = len(dict_list)
    array_of_floats = np.zeros((num_unique_keys, n), dtype=int)
    
    # Fill the array with values from the dictionaries
    for i, d in enumerate(dict_list):
        for j, key in enumerate(unique_keys):
            array_of_floats[j, i] = d.get(key, 0)  # Use 0 if the key is not present in the dictionary
    
    # Convert the set of unique keys back to a list
    unique_keys_list = np.array(list(unique_keys))
    
    # Print the result
    # print("Array of floats:")
    # print(array_of_floats)
    # print("\nList of unique dictionary keys:")
    # print(unique_keys_list)
    
    return unique_keys_list, array_of_floats



def Mix_mie(components_list):
    ncompounds = len(components_list)
    groups = []
    ind_comp = {}
    MM = np.zeros(ncompounds)
    for i in range( ncompounds ):
       groups += (list(components_list[i]['groups'].keys()))
       name = list(components_list[i]['name'])[0]
       ind_comp[name] = i
       if 'MM' in components_list[i]:
           MM[i] = components_list[i]['MM']
           
       
    groups = np.unique(groups)
    n_groups = len(groups)
    
    niki = np.zeros([n_groups,ncompounds])
    
    for i in range(n_groups):
        for j in range(ncompounds):
            g= groups[i]
            if g in components_list[j]['groups']:
                niki[i,j] = components_list[j]['groups'][g]
    
    
    return groups,niki,ind_comp,MM
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    