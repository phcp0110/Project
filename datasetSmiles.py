import pandas as pd
import numpy as np
from rdkit import Chem
from itertools import combinations
from sklearn.metrics import jaccard_score

### 1. Filtragem de moléculas com SMILES válidos ###
def filtrar_moleculas_com_smiles_validos(df, smiles_col="smiles"):
    df = df.copy()
    df = df[df[smiles_col].notnull()]
    df = df[df[smiles_col].str.strip() != ""]
    df = df[df[smiles_col].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
    return df.reset_index(drop=True)

### 2. Carregar fingerprints do NPClassifier ###
def carregar_npclassifier_fps(path_csv, id_col="ids"):
    df = pd.read_csv(path_csv)
    fp_cols = [col for col in df.columns if col.startswith("npclassifier_")]
    fps = df.set_index(id_col)[fp_cols]
    return fps.astype(int)

### 3. Calcular similaridade ###
def tanimoto(v1, v2):
    if len(v1) != len(v2):
        return np.nan
    return jaccard_score(v1, v2)

### 4. Criar dataset de pares com similaridade ###
def criar_dataset_com_pares(df_validas, smiles_col="smiles", id_col="ids"):
    pares = []
    for (i1, row1), (i2, row2) in combinations(df_validas.iterrows(), 2):
        pares.append({
            "mol1_id": row1[id_col],
            "mol1_smiles": row1[smiles_col],
            "mol2_id": row2[id_col],
            "mol2_smiles": row2[smiles_col],
        })
    return pd.DataFrame(pares)

### 5. Adicionar similaridade com base nos fingerprints ###
def adicionar_npclassifier_similaridade(df_pares, np_fp_dict):
    def calcular_sim(row):
        id1, id2 = row["mol1_id"], row["mol2_id"]
        fp1 = np_fp_dict.get(id1)
        fp2 = np_fp_dict.get(id2)
        if fp1 is None or fp2 is None:
            return np.nan
        return tanimoto(fp1, fp2)
    
    df_pares["NPClassifierFP"] = df_pares.apply(calcular_sim, axis=1)
    return df_pares

### === EXECUÇÃO === ###

# 1. Carrega dataset original e filtra moléculas com SMILES válidos
df_original = pd.read_csv("teu_dataset.csv")  # substitui pelo teu ficheiro real
df_validas = filtrar_moleculas_com_smiles_validos(df_original, smiles_col="smiles")

# 2. Carrega fingerprints do NPClassifier
df_fps = carregar_npclassifier_fps("NPClassifier_fp.csv")  # ficheiro com os fingerprints
fps_dict = {idx: row.values for idx, row in df_fps.iterrows()}

# 3. Gera todas as combinações de pares
df_pares = criar_dataset_com_pares(df_validas, smiles_col="smiles", id_col="ids")

# 4. Calcula a similaridade NPClassifierFP
df_final = adicionar_npclassifier_similaridade(df_pares, fps_dict)

# 5. Exporta resultados
df_validas.to_csv("moleculas_com_smiles_validos.csv", index=False)
df_final.to_csv("mock_dataset_com_NPClassifierFP.csv", index=False)

