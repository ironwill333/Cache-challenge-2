from rdkit import Chem
import numpy as np
import gzip
import rdkit
from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


score_list = []
smile_list = []
mol_list = []

for i in range(500):
    with gzip.open(f'docking_results/docked{i}.sdf.gz', 'rb') as fin:
        suppl = Chem.ForwardSDMolSupplier(fin)

        groups = defaultdict(list)

        for mol in suppl:
            groups[mol.GetProp("_Name")].append(mol)

        new_list = groups.values()

        for group in new_list:
            cur_score = max(mol.GetProp("CNN_VS") for mol in group)
            for mol in group:
                if cur_score == mol.GetProp("CNN_VS"):
                    mol_list.append(mol)


# Obtain numpy arrays of fingerprints and scaffolds
def fp_and_scaffold(mol_list) -> np.ndarray:
    fpSize = 2048
    fingerprints = np.zeros((len(mol_list), fpSize), dtype=bool)
    scaffold_groups = np.zeros(len(mol_list), dtype=int)
    gen = GetRDKitFPGenerator(minPath=1, maxPath=7, fpSize=fpSize)
    scaffold_dict = {}
    for i, mol in enumerate(mol_list):
        score_list.append(mol.GetProp("CNN_VS"))
        fingerprints[i, :] = gen.GetFingerprintAsNumPy(mol)
        scaffold = MurckoScaffoldSmiles(mol=mol)
        if scaffold not in scaffold_dict:
            scaffold_dict[scaffold] = len(scaffold_dict)
        scaffold_groups[i] = scaffold_dict[scaffold]
    return fingerprints, scaffold_groups

fingerprints, scaffold_groups = fp_and_scaffold(mol_list)
scores = np.array(score_list, dtype=float)

np.save('fingerprints.npy', fingerprints)
np.save('scores.npy', score_list)



# for i in range(20):
#     top_mol = 0
#     for j in range(9):   
#         cur_mol = suppl[mol_counter]
#         if j == 0:
#             top_mol = cur_mol
#             continue   
#         if cur_mol.GetProp("CNN_VS") > top_mol.GetProp("CNN_VS"):
#             top_mol = cur_mol
#         if j == 8:
#             mol_list.append(top_mol)
#         mol_counter += 1

# Get best pose of each molecule
# for i in range(200):
#     mol_list.append(suppl[i*9])