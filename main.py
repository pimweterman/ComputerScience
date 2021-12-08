import numpy as np
from ComputerScience import clean
from ComputerScience import convert_binary
from ComputerScience import minhash
from ComputerScience import lsh
from math import comb
import random
import spacy
from sklearn.cluster import KMeans
import re




def main():
    file_path = "Data/TVs-Clean.json"


    data_list, duplicates = clean(file_path)

    threshold = [x / 100 for x in range(5, 100, 5)]     #Change this to a range
    bootstraps = 5
    random.seed(0)

    results_path = "Results/"


    with open(results_path + "resultsExtension.csv", 'w') as out:
        out.write("t,comparisons,pq,pc,f1_star,f1\n")

    for t in threshold:
        print("t = ", t)

        #Initialize statistics
        results = np.zeros(5)

        for run in range(bootstraps):
            data_sample, duplicates_sample = bootstrap(data_list, duplicates)
            comparisons, pq, pc, f1_star, f1 = do_lsh(data_sample, duplicates_sample, t)
            results += np.array([comparisons, pq, pc, f1_star, f1])

        #Computer average statistics
        statistics = results / bootstraps

        with open(results_path + "resultsExtension.csv", 'a') as out:
            out.write(str(t))
            for stat in statistics:
                out.write("," + str(stat))
            out.write("\n")

def run(path, name):
    file_path = path
    file_name = name
    data_list, duplicates = clean(file_path)

    threshold = [x / 100 for x in range(5, 100, 5)]  # Change this to a range
    bootstraps = 5
    random.seed(0)

    results_path = "Results/"

    with open(results_path + file_name, 'w') as out:
        out.write("t,comparisons,pq,pc,f1_star,f1\n")

    for t in threshold:
        print("t = ", t)

        # Initialize statistics
        results = np.zeros(5)

        for run in range(bootstraps):
            data_sample, duplicates_sample = bootstrap(data_list, duplicates)
            comparisons, pq, pc, f1_star, f1 = do_lsh(data_sample, duplicates_sample, t)
            results += np.array([comparisons, pq, pc, f1_star, f1])

        # Computer average statistics
        statistics = results / bootstraps

        with open(results_path + file_name, 'a') as out:
            out.write(str(t))
            for stat in statistics:
                out.write("," + str(stat))




def do_lsh(data_list, duplicates, t):
    binary_vector = convert_binary(data_list)
    r = len(binary_vector)
    new_length = round(round(r / 2) / 100) * 100
    signature = minhash(binary_vector, new_length)
    candidates = lsh(signature, t)

    #Compute number of comparisons
    comparisons = np.sum(candidates) * 0.5
    comparisons_fraction = comparisons / comb(len(data_list), 2)

    #Compute matrix of correctly matched duplicates. Element (i,j) = 1
    # if elements i and j are duplicates and correctly classified by LSH
    correct = np.where(duplicates + candidates == 2, 1, 0)
    num_correct = np.sum(correct) * 0.5

    #Compute different performance metrix
    #Compute Pair Quality (PQ)
    pq = num_correct / comparisons

    #Compute Pair Completeness (PC)
    pc = num_correct / (np.sum(duplicates) * 0.5)

    #Computer F1-Measure
    f1_star = 2 * pq * pc / (pc + pq)

    #Cluster and compute F1-measure
    tp, precision = cluster(data_list, candidates, duplicates)
    recall = tp / (np.sum(duplicates) / 2)
    f1 = 2 * precision * recall / (precision + recall)

    return comparisons_fraction, pq, pc, f1_star, f1


def bootstrap(data_list, duplicates):

    #Compute different indices which are included in bootstrap
    indices = [random.randint(x, len(data_list) -1) for x in [0]*len(data_list)]

    #Get samples
    data_sample = [data_list[index] for index in indices]
    duplicates_sample = np.take(np.take(duplicates, indices, axis=0), indices, axis=1)
    return data_sample, duplicates_sample


def cluster(data_list, candidates, duplicates):


    nlp = spacy.load("en_core_web_md", disable=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat'])

    titles = []
    duplicate_vec = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if candidates[i, j] == 1:
                # Find model words in the titles.
                mw_i = re.findall(
                    "([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*)",
                    data_list[i]["title"])
                mw_title_i = ""
                for match in mw_i:
                    if mw_i[0] != '':
                        mw_title_i += match[0]
                    else:
                        mw_title_i += match[1]
                mw_j = re.findall(
                    "([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*)",
                    data_list[j]["title"])
                mw_title_j = ""
                for match in mw_j:
                    if mw_j[0] != '':
                        mw_title_j += match[0]
                    else:
                        mw_title_j += match[1]
                titles.append([mw_title_i, mw_title_j])

                if duplicates[i, j] == 1:
                    # (i, j) true duplicates.
                    duplicate_vec.append(1)
                else:
                    # (i, j) not duplicates.
                    duplicate_vec.append(0)

    titles = np.array(titles)

    def similarity(x):

        title_i = nlp(str(x[0]))
        title_j = nlp(str(x[1]))
        return title_i.similarity(title_j)

    # Compute similarities of candidate pair product titles using spaCy.
    similarities = np.apply_along_axis(similarity, 1, titles)

    duplicate_vec = np.array(duplicate_vec)

    # Cluster using K-means.
    kmeans = KMeans(n_clusters=2, random_state=random.randint(1, 100000)).fit(similarities.reshape(-1, 1))
    labels = kmeans.labels_
    mean_0 = np.sum(np.where(labels == 0, similarities, 0)) / np.count_nonzero(labels == 0)
    mean_1 = np.sum(np.where(labels == 1, similarities, 0)) / np.count_nonzero(labels == 1)

    # Check which pairs are more similar on average, and declare these to be the duplicates.
    predictions = np.where(labels == 0, 1, 0) if mean_0 > mean_1 else np.where(labels == 1, 1, 0)

    # Compute precision.
    tp = np.sum(np.where(predictions == duplicate_vec, predictions, 0))
    precision = tp / np.sum(predictions)

    return tp, precision


main()
