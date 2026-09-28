import numpy as np
import torch
from scipy.stats import mode


def LabelPred(prob, class_0_th, class_1_th):
    # 动态调整置信度阈值。
    if prob < class_0_th:
        return 0
    if prob > class_1_th:
        return 1
    else:
        return -1


def MajorityVote(row):
    row = np.array(row)
    if np.all(row == -1):
        return -1
    else:
        m = mode(row[row != -1])
        if isinstance(m.mode, np.ndarray):
            return m.mode[0]
        else:
            return m.mode


def select_data(index_dict_list, class_0_th=0.3, class_1_th=0.7, budget=0.7, K=20):
    # 引入动态权重和一致性正则化逻辑。
    fet_dict = {}
    label_dict = {}
    fet_list_flag = False

    # 逐步生成特征和标签
    for index_dict in index_dict_list:
        for index in sorted(index_dict):
            if not fet_list_flag:
                fet_dict[index] = index_dict[index]["fet"]

            if index in label_dict:
                label_dict[index].append(
                    LabelPred(index_dict[index]["prob"], class_0_th=class_0_th, class_1_th=class_1_th)
                )
            else:
                label_dict[index] = [LabelPred(index_dict[index]["prob"], class_0_th=class_0_th, class_1_th=class_1_th)]
        fet_list_flag = True

    # 构建特征和标签数组
    fetArray = np.vstack([fet_dict[i] for i in range(len(label_dict))])
    labelArray = np.vstack([MajorityVote(label_dict[i]) for i in range(len(label_dict))])
    fetArray = torch.tensor(fetArray)
    labelArray = torch.tensor(labelArray)

    # 获取非弃权标签
    non_abstrain = torch.tensor(labelArray != -1).squeeze(1)

    if sum(non_abstrain) > 10:  # 至少 10 个样本
        non_abstrain_label = labelArray[non_abstrain].squeeze(1)
        non_abstain_fet = fetArray[non_abstrain]
        non_abstrain_index = torch.arange(len(labelArray))[non_abstrain]

        # 动态选择数据，加入伪标签权重
        select_index = get_cutstat_inds(non_abstain_fet, non_abstrain_label, coverage=budget, K=K)
        select_index = torch.tensor(select_index)

        final_label = torch.index_select(non_abstrain_label, 0, select_index)
        final_index = torch.index_select(non_abstrain_index, 0, select_index)

        return final_index, final_label
    else:
        return [], []


def get_cutstat_inds(features, labels, coverage=0.5, K=20, device='cpu'):
    # 结合 SALFL 思路，引入动态权重。
    pairwise_dists = torch.cdist(features, features, p=2).to('cpu')
    N = labels.shape[0]
    dists_sorted = torch.argsort(pairwise_dists)
    neighbors = dists_sorted[:, :K]
    dists_nn = pairwise_dists[torch.arange(N)[:, None], neighbors]
    weights = 1 / (1 + dists_nn)

    # 动态权重计算
    neighbors = neighbors.to(device)
    dists_nn = dists_nn.to(device)
    weights = weights.to(device)
    cut_vals = (labels[:, None] != labels[None, :]).long()
    cut_neighbors = cut_vals[torch.arange(N)[:, None], neighbors]

    Jp = (weights * cut_neighbors).sum(dim=1)
    weak_counts = torch.bincount(labels)
    weak_pct = weak_counts / weak_counts.sum()
    prior_probs = weak_pct[labels.long()]

    mu_vals = (1 - prior_probs) * weights.sum(dim=1)
    sigma_vals = prior_probs * (1 - prior_probs) * torch.pow(weights, 2).sum(dim=1)
    sigma_vals = torch.sqrt(sigma_vals)

    normalized = (Jp - mu_vals) / sigma_vals
    normalized = normalized.cpu()

    # 按置信度排序并选择样本
    inds_sorted = torch.argsort(normalized)
    N_select = int(coverage * N)
    conf_inds = inds_sorted[:N_select]
    conf_inds = list(set(conf_inds.tolist()))

    return conf_inds
