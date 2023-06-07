#!/usr/bin/env python3

import time
import random
import sys
from pathlib import Path
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps

# show full results
np.set_printoptions(threshold=sys.maxsize)


# Reading the input CSV file.

ligands_df = pd.read_csv("PFAS74.csv" , index_col=0 )
print(ligands_df.head())



# Creating molecules and storing in an array
molecules = []

"""Let's fetch the smiles from the input file and store in molecules array
        We have used '_' because we don't want any other column.
        If you want to fetch index and any other column, then replace '_' with 
            index and write column names after a ','.
"""

for _, smiles in ligands_df[[ "MS_READY_SMILES"]].itertuples():
    molecules.append((Chem.MolFromSmiles(smiles)))
molecules[:73]


# Creating fingerprints for all molecules

rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)
fgrps = [rdkit_gen.GetFingerprint(mol) for mol in molecules]


# Calculating number of fingerprints
nfgrps = len(fgrps)
print("Number of fingerprints:", nfgrps)


# Defining a function to calculate similarities among the molecules
def pairwise_similarity(fingerprints_list):
    
    global similarities

    similarities = np.zeros((nfgrps, nfgrps))

    for i in range(1, nfgrps):
            similarity = DataStructs.BulkTanimotoSimilarity(fgrps[i], fgrps[:i])
            similarities[i, :i] = similarity
            similarities[:i, i] = similarity

    return similarities


# Calculating similarities of molecules
pairwise_similarity(fgrps)
tri_lower_diag = np.tril(similarities, k=0)


# writing similarities to file

np.savetxt("similarities.csv", similarities, delimiter = ",")

np.savetxt("similarities_lower_tri.csv", tri_lower_diag, delimiter = ",")

