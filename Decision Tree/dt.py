# 0. Libarary
import pandas as pd
import numpy as np
import sys

# 1. Data Load
data = pd.read_csv(sys.argv[1], delimiter='\t') # Load train
test = pd.read_csv(sys.argv[2], delimiter='\t') # Load test
answer = pd.read_csv(sys.argv[3], delimiter='\t') # Load answer

# X (attributes)
features = data.iloc[:,:-1]

# Y(target)
target = data.iloc[:,-1:]

# 2. Entropy Function
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts = True) # Number of total data 
    print(elements) # elements of targets e.g. ['yes', 'no'] / ['acc', 'good', 'unacc', 'vgood']
    print(counts) # Count of each targets. e.g. [5, 9] / [302 , 52, 975, 53]
    entropy = -np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy # Entropy of total data.
                      
print('H(x) = ', entropy(target))


# 3. InfoGain Function
def InfoGain(data, split_attribute_name, target_name):

    # Calculate total Entropy.
    total_entropy = round(entropy(data[target_name]), 5) 

    # Calculate split_attribute Entropy.
    vals, counts = np.unique(data[split_attribute_name],return_counts=True)
    
    Splited_Entropy = np.sum([(counts[i]/np.sum(counts))*
                               entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name])
                               for i in range(len(vals))])
    print('H(', split_attribute_name, ') = ', round(Splited_Entropy, 5))

    # Caculate Infomation Gain
    Information_Gain = total_entropy - Splited_Entropy # Infomation Gain = Total Entropy - each split attribute Entropy. 
    return Information_Gain


# 4. ID3 Algorithm
def ID3(data, original_data, features, target_attribute_name, parent_node_class = None):

    # Define Standard of suspend spliting

    # 1. If Y is single value, return corresponding Y(target) 
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # 2. If there is no data(length is zero), return Y which has maximum value (using argmax) in original data.
    elif len(data)==0:
        return np.unique(original_data[target_attribute_name])\
               [np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]

    # 3. If there is no X, return Y of parent node.
    elif len(features) ==0:
        return parent_node_class

    # Growth Tree
    else:
        # Define Y of parent node.
        parent_node_class = np.unique(data[target_attribute_name])\
                            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # Select attribute which is appropriate for spliting. (target: Yes or No)
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        
        # Having maximum InfoGain = best_feature_index(Select the highest Info Gain)
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Generate Tree Structure.
        tree = {best_feature:{}}

        # Exclude already used best feature.
        features = [i for i in features if i != best_feature] # Except a best_feature = features

        # Growth Pruning.
        for value in np.unique(data[best_feature]):
            # Split data. But if data has NA, using dropna to remove row, columns.
            sub_data = data.where(data[best_feature] == value).dropna()

            # Call ID3 function(recursive) 
            subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class) 
            tree[best_feature][value] = subtree
      
        return(best_feature, tree)

# Completed tree
tree = ID3(data, data, data.columns[:-1], data.columns[-1:])


# 5. Classification
def classify(tree, input):
    if tree in np.unique(target): # If there is target columns in tree, return tree
        return tree
    try: # Make a Exception.
        attribute, subtree_dict = tree # attribute = best_feature , subtree = tree(recursive)
    except:
        print("Error")
        print(tree)

    subtree_key = input.get(attribute) # key of subtree. e.g. dt_train ) First key = age , Second key = '<=30', '31..40', '>40' ...

    if subtree_key not in subtree_dict[attribute]: # When the subtree corresponding to key does not exist, use the None subtree.
        subtree_key = list(subtree_dict[attribute].keys())[0]
    subtree = subtree_dict[attribute][subtree_key]
    
    return classify(subtree, input) # recursive


if __name__ == '__main__':
    # Save arguments list
    argv = sys.argv 

    # Classification Prediction 
    result = []
    for i in range(len(test)):
        data = test.loc[i].to_dict() # Read row one by one. And make a dict.
        result.append(classify(tree, data)) # Append the prediction result in list.

    # Compare the answer and print the score.
    score=0
    for j in range(len(answer)): # Compare my prediction and answer row by row.
        answer_target = answer.iloc[:,-1:] 
        if result[j] == answer_target.values[j]:
            score+=1
    print('%d / %d' %(score, len(answer))) # the number of my correct prediction
    print('정답률: %.2f%%' %((score/len(answer))*100)) # Percentage of Answer

    target_answer = pd.DataFrame(result, columns=target.columns) # Convert my prediction to dataframe.
    output = pd.concat([test, target_answer], axis=1) # Concat test data and my prediction.

    output.to_csv(sys.argv[4], index=True, sep='\t') # Save result as dt_result.txt, dt_result1.txt
  
