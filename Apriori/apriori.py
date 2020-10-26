import pandas as pd
import numpy as np

import sys
import itertools
from collections import defaultdict

# 1) Read input.txt File
def load_data(): 
    global item_list # This is initial list which contain all of itemsets.
    item_list = list()
    
    with open('input.txt', 'r') as f: 
        lines = f.read().split('\n') # Read by a line-break
        for line in lines: 
            line = line.split('\t') # Read by a space
            item_list.append(line) 
    return item_list


# 2) Initial Frequent Itemset
def init_freq_set():
    global item_list
    localSet = defaultdict(int) # defaultdict : default of dictionary    
    
    for line in item_list: # Count items
        for item in line:
            if item not in localSet:
                localSet[item] = 1   
            else:
                localSet[item] += 1 
    return pruning_min_sup(localSet)


# 3) Pruning Under Minimum Support
def pruning_min_sup(candidate):
    global item_list    
    cnt = min_support * len(item_list) # min_support = cnt/len(item_list)   
    _itemSet = {key: candidate[key] for key in candidate.keys() 
                if candidate[key] >= cnt}
#   print(_itemSet)

    if len(_itemSet) < 1: # Terminate when there is no more _itemSet
        print('Terminate') 
        return _itemSet
    else:        
        return _itemSet # Insert itemSet which is bigger than minimum support.
    

# 4) Self-Joining Frequent itemset
def self_joining(length, prev_freq_set):
    join_list = list()
    
    if length == 2: #two candidates
        for item in itertools.combinations(prev_freq_set, length): # union of previous frequent itemsets. 
            join_list.append(item)
        return list(map(set, join_list))

    else: #over three candidates
        for item_set in prev_freq_set: # Prevent duplicated frequent items.
            for item in item_set:
                if item not in join_list:
                    join_list.append(item)
                    
        for item in itertools.combinations(join_list, length): # union of frequent itemsets over three candidates. 
            join_list.append(item)
        candidate = list(map(set, join_list)) #Make a set for candidate
        
        return candidate

# 5) Pruning 
def pruning(length, prev_freq_set, candidate):
    global item_list
    _itemSet = dict()

    if length == 2 : #two candidates
        temp = list() 
        for item in prev_freq_set:
            temp.append(list([item,])) # For append two digits itemset.
        prev_freq_set = temp

    else: #over three candidates
        join_list = list()
        for item in prev_freq_set:
            join_list.append(set(item))
        prev_freq_set = join_list  
     
    for item_set in candidate: 
        cnt = 0 # compare frequent set and previous frequent set.
        for item in list(itertools.combinations(item_set, length - 1)):
            if length == 2:
                item = list(item) 
            else:
                item = set(item)

            if item not in prev_freq_set: 
                break
            cnt = cnt + 1

        if cnt == length:
            _itemSet[tuple(item_set)] = 0 

    for key in _itemSet.keys(): # For next frequent set.
        for line in item_list:
            if set(key) <= set(line): 
                _itemSet[key] = _itemSet[key] + 1

    return pruning_min_sup(_itemSet)

# 6) association_rule : calculate support, confidence about all frequent-itemset.
def association_rule(length, _itemSet):
    for item_set, freq in _itemSet.items():
        frequent_set_len = length        
        while frequent_set_len > 1: #iterate when no frequent itemset
            combi = list(itertools.combinations(item_set, frequent_set_len-1))
            for item in combi:
                item = set(item)    
                remain = set(item_set) - set(item) #difference for counterpart combi
                support = freq / len(item_list) * 100 
                
                cnt_item = 0
                for line in item_list:
                    if set(line) >= item:
                        cnt_item = cnt_item + 1
            
                confidence = freq / cnt_item * 100 
                
                # string -> int
                item = set(map(int, item))
                remain = set(map(int, remain))

                # print format : rounded to two decimal places
                line = str(item) + '\t' + str(remain) + '\t' + str('%.2f' % round(support, 2)) + '\t' + str('%.2f' % round(confidence, 2)) + '\n'
                save_result(line)
                print(line)
            frequent_set_len -= 1

# 7) Save output
def save_result(line):
    with open(sys.argv[3], 'a') as f: # Save fourth argument name.   
            f.write(line)


if __name__ == '__main__':

    argv = sys.argv 
    min_support = float(argv[1])/100 
    # min_support = 0.02
    output = argv[3]

    load_data() 
    _itemSet = ['',]
    _itemSet.append(init_freq_set())
    
    length = 2
    while True:
     
        prev_freq_set = list(_itemSet[length - 1].keys())   
        candidate = self_joining(length, prev_freq_set)
         
        if len(candidate) == 0: #Terminate when there is no more candidate
            exit()
        
        candidate = pruning(length, prev_freq_set, candidate)
        association_rule(length, candidate)
        if candidate == 0: #Terminate when there is no more candidate
            exit()
        else:
            _itemSet.append(candidate)
            length = length + 1           









