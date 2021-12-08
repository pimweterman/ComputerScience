import json
import re
import random
from random import shuffle
from HelpFunctions import prime
import numpy as np
import sys
from sympy import nextprime





#Opening json file
file_path = "Data/TVs-all-merged.json"
file_path_clean = "Data/TVs-Clean.json"

def clean(file_path):
    file = open(file_path)
    data = json.load(file)

    #Declare common representations to be replaced by the last value of each list
    inch = ["Inch", "inches", "\"", "-inch", " inch", "inch"]
    hz = ["Hertz", "hertz", "Hz", "HZ", " hz", "-hz", "hz"]
    lbs = [" pounds", "pounds", "lb ", " lbs", "lbs"]
    year = [" year limited", " years limited", " years", " year", " Year", "Year", "year"]
    mw = [" mW", "mW"]
    to_replace = [inch, hz, lbs, year, mw]
    replacements = dict()
    for replace_list in to_replace:
        replacement = replace_list[-1]
        values = replace_list[0:-1]
        for value in values:
            replacements[value] = replacement

    #Clean the data
    clean_list = []
    for model in data:
        for occurence in data[model]:
            #clean title
            for value in replacements:
                occurence["title"] = occurence["title"].replace(value, replacements[value])

            #clean rest of features
            features = occurence["featuresMap"]
            for key in features:
                for value in replacements:
                    features[key] = features[key].replace(value, replacements[value])
            clean_list.append(occurence)

    #Shuffle the item list
    shuffle(clean_list)

    #Compute binary matrix of duplicates for testing
    #element (i, j) is 1 if i and j are duplicates, 0 otherwise. Symmetric matrix
    duplicates = np.zeros((len(clean_list), len(clean_list)))
    for i in range(len(clean_list)):
        modeli = clean_list[i]["modelID"]
        for j in range(i + 1, len(clean_list)):
            modelj = clean_list[j]["modelID"]
            if modeli == modelj:
                duplicates[i][j] = 1
                duplicates[j][i] = 1
    return clean_list, duplicates.astype(int)

#Convert the data in binary vectors

def convert_binary(clean_list):

    #We keep all model words as keys in a dictionary, where its value
    # is the corresponding row in the binary vector product representation

    model_words = dict()
    binary_vector = []

    # Loop through all the items to find the model words
    for i in range(len(clean_list)):
        item = clean_list[i]
        # Find model words in the title
        mw_title = re.findall("([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*)", item["title"])
        item_mw = mw_title
        #HIER EVENTUEEL NOG STUK BIJDOEN

        # Find model words in key value pairs
        features = item["featuresMap"]
        for key in features:
            value = features[key]
            mw_keyvalue = re.findall("(?:(?:(^[0-9]+(?:\.[0-9]+))[a-zA-Z]+$)|(^[0-9](?:\.[0-9]+)$))", value)
            for word in mw_keyvalue:
                for group in word:
                    if group != "":
                        item_mw.append(group)

        #Loop through all identified model words and update the binary vector product representations
        for word in item_mw:
            if word in model_words:
                #Set index for model word to 1
                row = model_words[word]
                binary_vector[row][i] = 1
            else:
                #Add model word to the binary vector and set index to one
                binary_vector.append([0] * len(clean_list))
                binary_vector[len(binary_vector) - 1][i] = 1

                #Add model word to the dictionary
                model_words[word] = len(binary_vector) - 1

    length = len(binary_vector)
    columns =len(binary_vector[0])
    model_words_final = item_mw
    return binary_vector


def minhash(binary_vector, n):

    random.seed(1)
    r = len(binary_vector)          # number of model words
    c = len(binary_vector[0])       # number of products/items
    binary_vector = np.array(binary_vector)

    #Find k (use Help function)
    k = prime(r-1)

    # Generate n random hash functions
    hash_params = np.empty((n, 2))
    for i in range(n):
        a = random.randint(1, k - 1)
        b = random.randint(1, k - 1)
        hash_params[i, 0] = a
        hash_params[i, 1] = b

    #Initialize signature matrix
    signature = np.full((n, c), np.inf)

    # Loop through binary vector matrix
    for row in range(1, r + 1):
        # Compute each of the n random hashes once for each row
        e = np.ones(n)
        row_vec = np.full(n, row)
        x = np.stack((e, row_vec), axis=1)
        row_hash = np.sum(hash_params * x, axis=1) % k

        for i in range(n):
            # Update col j if and only if it contains a 1 and its current value is larger
            # than the hash value for the signature matrix row i
            update = np.where(binary_vector[row - 1] == 0, np.inf, row_hash[i])
            signature[i] = np.where(update < signature[i], row_hash[i], signature[i])
    return signature.astype(int)


# Method to implement lsh based on the obtained minHash signature matrix
def lsh(signature, threshold):

    n = len(signature)

    # Compute approximate number of bands and rows from threshold t, using n = r * b
    # and t is approximately (1/b)^(1/r)
    b_opt = 1
    r_opt = 1
    opt = 1

    for r in range(1, n+1):
        for b in range(1, n+1):
            #Check if is a valid pair
            if r*b == n:
                approx_t = (1/b)**(1/r)
                #Get the best t
                if abs(approx_t - threshold) < abs(opt - threshold):
                    opt = approx_t
                    r_opt = r
                    b_opt = b

    # Create array with zeros for the candidate pairs
    # signatureM[0] = 1624 (#items)
    candidates = np.zeros((len(signature[0]), len(signature[0])))

    for band in range(b_opt):
        buckets = dict()
        start_row = band * r_opt        # This row is included
        end_row = (band+1) * r_opt      # This row is excluded
        strings = ["".join(signature[start_row:end_row, column].astype(str)) for column in range(len(signature[0]))]
        ints = [int(string) for string in strings]
        hashes = [integer % sys.maxsize for integer in ints]

        # Add all the hashes of the itmes to the correct bucket
        for item in range(len(hashes)):
            hash_value = hashes[item]
            if hash_value in buckets:

                #All items in this bucket are possible duplicates
                for candidate in buckets[hash_value]:
                    candidates[candidate, item] = 1
                    candidates[item, candidate] = 1
                buckets[hash_value].append(item)
            else:
                buckets[hash_value] = [item]
    return candidates.astype(int)