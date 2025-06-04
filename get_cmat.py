from tqdm import tqdm
import numpy as np


def manhattan_distance(x, y):
    """
    Compute Manhattan distance between two vectors.
    Parameters:
        x, y : array-like
            Input vectors (must be the same length).
    Returns:
        float : Distance score.
    """
    x = np.array(x)
    y = np.array(y)
 
    distance = np.sum(np.abs(x - y))
 
    return distance

def compute_all_distances(dataset):
    """
    Compute all pairwise similarities in the dataset.
    Parameters:
        dataset : list of lists
            Each inner list represents a sample.
    Returns:
        list of tuples : Each tuple contains two indices and their similarity score.
    """
    fp = dataset.X.copy()
    
# Preallocate vector of correct size
    n_samples = fp.shape[0]
    n_comps = int((n_samples ** 2 - n_samples) / 2)
    distances = np.zeros((1, n_comps,), dtype=np.float32)

    # Start pairwise similarity counter
    count = 0

    # Loop over all unique pairs (skip redundant comparisons)
    for i in tqdm(range(n_samples - 1), desc="Computing distances"):
        for j in range(i + 1, n_samples):
            # Calculate similarity as 1 - Jaccard distance
            distances[0, count] = manhattan_distance(fp[i, :], fp[j, :])
            count += 1

    return distances

import numpy as np

def normalize_similarity_matrix(similarity_matrix):
    """
    Normalize the similarity matrix to a range of [0, 1].
    Parameters:
        similarity_matrix : numpy array
            The similarity matrix to normalize.
    Returns:
        numpy array : Normalized similarity matrix.
    """
    max_val = np.max(similarity_matrix)
    # min_val = np.min(similarity_matrix)
    min_val = 0
 
    # Handle the case where max_val and min_val are the same to avoid division by zero
    if max_val == min_val:
        return np.ones_like(similarity_matrix) * 0.5  # or any other default value
 
    normalized_matrix = (similarity_matrix - min_val) / (max_val - min_val)
 
    return 1 - normalized_matrix
 

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle 


def generate_mst(similarities_matrix, labels):

    # Step 2: Convert the Correlation Matrix to a Distance Matrix
    distance_matrix = 1 - similarities_matrix

    # Create a graph from the distance matrix
    G = nx.Graph()
    num_nodes = distance_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, weight=distance_matrix[i, j])

    # Compute the minimum spanning tree without inversion
    mst_without_inversion = nx.minimum_spanning_tree(G)

    # Set the labels for the nodes in the graph
    for node in mst_without_inversion.nodes():
        if node in labels:
            mst_without_inversion.nodes[node]['label'] = labels[node]
        else:
            mst_without_inversion.nodes[node]['label'] = str(node)

    return mst_without_inversion

def fingerprint_pipeline(dataset, fingerprints, labels):
    
    ''' 
dataset: 
fingerprints: lista de featurizes 
labels: dicionario com os métodos 
random_state: random seed 

'''
    # Importações 
    import numpy as np

    similarity_matrices = [] #Lista onde vamos guardar os vetores de similaridade
    i=0
    # Aplicar uma a uma asfingerprints ao dataset
    for featurizer in fingerprints:
        featurizer.featurize(dataset, inplace=True)
        dataset.to_csv(f"{featurizer.__class__.__name__}_fp.csv")

        distances = compute_all_distances(dataset) # Calcula todas as similaridades entre as moleculas 
        similarities = normalize_similarity_matrix(distances) # Normaliza esses valores

        similarity_matrices.append(similarities) # Guarda as similaridades de cada método

    similarities = np.concatenate(similarity_matrices, axis=0) # Junta tudo numa matriz
        
    
    correlation_matrix = np.corrcoef(similarities) # Calcula a correlation entre os métodos de fingerprint, quão semelhantes são os vetores gerados por cada método
    with open("correlation_matrix.pkl","wb") as f:
        pickle.dump(correlation_matrix, f)
         # Chama a função generate_mst que vai usar a correlation matrix para gerar a MST
    
    with open("similarities.pkl", "wb") as f:
        pickle.dump(similarities, f)
    
    mst = generate_mst(similarities, labels)   
    with open("mst.pkl","wb") as f:
        pickle.dump(mst, f)
    

    return correlation_matrix, mst

def violin_plot(similarity_matrix):
    
    labels = [
    "",
    "NPClassifierFP",
    "Biosynfoni",
    "NeuralNPFP",
    "MHFP", 
    "MorganFingerprint",
    "NPBERT"
    ]

    fig, ax = plt.subplots()
    ax.set_xticklabels(labels=labels, )
    ax.set_xlabel("Fingerprint Method")           
    ax.set_ylabel("Similarity Score")   

    for i in range(0,similarity_matrix.shape[0]):
        ax.violinplot(dataset=similarity_matrix[i],positions=[i])
    plt.xticks(rotation=45)
    plt.title("Distribution of Similarity Scores by Fingerprint")
    plt.tight_layout()

    plt.savefig("violinplot_similarity.png", dpi=300)
    


if __name__ == "__main__":
    from deepmol.loaders import CSVLoader
    import pandas as pd
    import numpy as np
    from deepmol.compound_featurization import NPClassifierFP, BiosynfoniKeys, NeuralNPFP, MHFP, MorganFingerprint
    from deepmol.tokenizers import NPBERTTokenizer
    import os
    from deepmol.compound_featurization import LLM
    from transformers import BertConfig, BertModel

# Load data from CSV file
    loader = CSVLoader(dataset_path='amostras_30000.csv',
                   smiles_field='smiles',
                   id_field='ids',
                   mode='auto')
# create the dataset
    csv_dataset = loader.create_dataset(sep=',', header=0)
    fingerprint_pipeline(
        csv_dataset,                     # O dataset carregado com SMILES
        fingerprints=[NPClassifierFP(), BiosynfoniKeys(), NeuralNPFP(), MHFP(), MorganFingerprint(), LLM(model_path="NPBERT", model=BertModel, config_class=BertConfig,
                          tokenizer=NPBERTTokenizer(vocab_file=os.path.join("NPBERT", "vocab.txt")), device="cuda:0")],
        labels={0: "NPClassifierFP",
                1: "BiosynfoniKeys",
                2: "NeuralNPFP",
                3: "MHFP", 
                4: "MorganFingerprint",
                5: "NPBert"
        }
                )
    
    with open("similarities.pkl", "rb") as f:
        similarity_matrix = pickle.load(f)
    violin_plot(similarity_matrix)