import torch
results_data = {"TCGA" : -1, "CCLE" : -2}
cosine_flag = True
ccle_only = False
############ Subset Selection #########
subset_selection_flag = True

#seed = 1221 327 1984 2000

seed = 1795
folder_name = "logs/explain_2"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eff_drug_list = ['cis','dox','sor','sun','tem']
test_data_index = results_data["TCGA"]
basis_drug_list = ['fu', 'tem', 'gem', 'cis', 'sor','sun', 'dox', 'tam', 'pac', 'car', 'Cetuximab', 'Methotrexate', 'Topotecan', 'Erlotinib', 'Irinotecan', 'Bicalutamide', 'Temsirolimus', 'Oxaliplatin', 'Docetaxel', 'Etoposide']
