import os
import torch.autograd as autograd
from itertools import chain
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit.Chem import AllChem, DataStructs
sys.path.append("../")
from config import param_config
from src.dsn_basis_ae_final import DSNBasisAE
from src.evaluation_utils import *
from src.mlp import MLP
from src.train_code_base import eval_basis_dsnae_epoch, basis_dsn_ae_train_step
from collections import OrderedDict
import numpy as np
from sklearn.decomposition import PCA
from rdkit import Chem
import pandas as pd

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def critic_dsn_train_step(critic, s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, history, scheduler=None,
                          clip=None, gp=None):
    critic.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.eval()
    t_dsnae.eval()
    critic.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)
    # 训练时不再使用final_code而是使用z_reparam
    s_code = s_dsnae.encode(s_x)[0]
    t_code = t_dsnae.encode(t_x)[0]

    loss = torch.mean(critic(t_code)) - torch.mean(critic(s_code))

    if gp is not None:
        gradient_penalty = compute_gradient_penalty(critic,
                                                    real_samples=s_code,
                                                    fake_samples=t_code,
                                                    device=device)
        loss = loss + gp * gradient_penalty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if clip is not None:
        for p in critic.parameters():
            p.data.clamp_(-clip, clip)
    if scheduler is not None:
        scheduler.step()

    history['critic_loss'].append(loss.cpu().detach().item())

    return history


def gan_dsn_gen_train_step(critic, s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, alpha, history,
                           scheduler=None):
    critic.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    critic.eval()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    s_label = s_batch[1].to(device)
    t_x = t_batch[0].to(device)
    t_label = t_batch[1].to(device)

    t_code = t_dsnae.encode(t_x)[0]
    optimizer.zero_grad()
    gen_loss = -torch.mean(critic(t_code))
    print("Before calling forward in train_fn...")
    s_loss_dict = s_dsnae.loss_function(s_x, s_target)  # 这里应该调用了 forward
    print("After calling forward in train_fn...")
    t_loss_dict = t_dsnae.loss_function(t_x, t_label)
    recons_loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss = recons_loss + alpha * gen_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    history['gen_loss'].append(gen_loss.cpu().detach().item())

    return history


def train_drug_adv(**kwargs):
    drug_names = [
        '5-Fluorouracil (Fu)', 'Temozolomide (Tem)', 'Sorafenib (Sor)', 'Gemcitabine (Gem)', 'Cisplatin (Cis)',
        'Sunitinib', 'Doxorubicin', 'Tamoxifen', 'Paclitaxel', 'Carmustine', 'Cetuximab', 'Methotrexate',
        'Topotecan', 'Erlotinib', 'Irinotecan', 'Bicalutamide', 'Temsirolimus', 'Oxaliplatin', 'Docetaxel', 'Etoposide'
    ]
    smiles = [
        'FC1=CN(C(=O)N1)C(=O)O',  # 5-Fluorouracil (Fu)
        'CN1C(=O)NC(=O)N1C2=CC=CC=C2',  # Temozolomide (Tem)
        'CC1=CC(=C(C(=C1)C(=O)O)C(=O)O)C2=CC=CC=C2',  # Sorafenib (Sor)
        'CN1C=NC(=O)N1C2=CC=CC=C2',  # Gemcitabine (Gem)
        'Cl[Pt]Cl',  # Cisplatin (Cis)
        'CC1=CC(=C(C(=C1)C(=O)O)C(=O)O)C2=CC=CC=C2',  # Sunitinib
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Doxorubicin
        'CC1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Tamoxifen
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Paclitaxel
        'CC1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Carmustine
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Cetuximab
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Methotrexate
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Topotecan
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Erlotinib
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Irinotecan
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Bicalutamide
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Temsirolimus
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Oxaliplatin
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3',  # Docetaxel
        'C1=CC2=C(C(=C1)C(=O)O)C(=C2C(=O)O)C3=CC=CC=C3'  # Etoposide
    ]

    smiles_df = pd.DataFrame({'Drug': drug_names, 'SMILES': smiles})

    def smiles_to_fingerprint(smiles, nBits=128):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nBits, dtype=int)

        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
        arr = np.zeros((nBits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return arr

    smiles_df['ECFP1'] = smiles_df['SMILES'].apply(lambda x: smiles_to_fingerprint(x, nBits=128))
    smiles_df['ECFP2'] = smiles_df['SMILES'].apply(lambda x: smiles_to_fingerprint(x, nBits=128))
    smiles_df['ECFP3'] = smiles_df['SMILES'].apply(lambda x: smiles_to_fingerprint(x, nBits=128))

    ecfp_embeddings1 = np.vstack(smiles_df['ECFP1'].apply(lambda x: x.astype(np.int32)).values)
    ecfp_embeddings2 = np.vstack(smiles_df['ECFP2'].apply(lambda x: x.astype(np.int32)).values)
    ecfp_embeddings3 = np.vstack(smiles_df['ECFP3'].apply(lambda x: x.astype(np.int32)).values)

    ecfp_tensor1 = torch.tensor(ecfp_embeddings1, dtype=torch.float32).to(kwargs['device'])
    ecfp_tensor2 = torch.tensor(ecfp_embeddings2, dtype=torch.float32).to(kwargs['device'])
    ecfp_tensor3 = torch.tensor(ecfp_embeddings3, dtype=torch.float32).to(kwargs['device'])

    mlp = MLP(input_dim=128, output_dim=128, hidden_dims=[128, 128], dop=kwargs['dop']).to(kwargs['device'])
    ecfp_out1 = mlp(ecfp_tensor1)
    ecfp_out2 = mlp(ecfp_tensor2)
    ecfp_out3 = mlp(ecfp_tensor3)

    mlp_for_dim_reduction = MLP(input_dim=384, output_dim=128, hidden_dims=[128, 128], dop=kwargs['dop']).to(kwargs['device'])
    combined_embeddings = torch.cat((ecfp_out1, ecfp_out2, ecfp_out3), dim=1)
    final_embeddings = mlp_for_dim_reduction(combined_embeddings)

    return final_embeddings


def train_code_adv(s_dataloaders, t_dataloaders, ccle_only, drug_dim, cosine_flag, **kwargs):
    s_train_dataloader = s_dataloaders[0]
    s_test_dataloader = s_dataloaders[1]
    t_train_dataloader = t_dataloaders[0]
    t_test_dataloader = t_dataloaders[1]

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])

    # basis_vec = torch.nn.Embedding(drug_dim, kwargs['latent_dim']).to(kwargs['device'])
    final_embeddings=train_drug_adv(**kwargs)
    basis_vec = torch.nn.Parameter(torch.tensor(final_embeddings, requires_grad=True)).to(kwargs['device'])

    inv_temp = kwargs['inv_temp']

    shared_decoder = MLP(input_dim=kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNBasisAE(shared_encoder=shared_encoder,
                         decoder=shared_decoder,
                         basis_vec=basis_vec,
                         input_dim=kwargs['input_dim'],
                         latent_dim=kwargs['latent_dim'],
                         testing_drug_len=kwargs['testing_drug_len'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         inv_temp=kwargs['inv_temp'],
                         dop=kwargs['dop'],
                         num_geo_layer=kwargs['num_geo_layer'],
                         cosine_flag=cosine_flag,
                         cns_basis_label_loss=True,
                         norm_flag=kwargs['norm_flag'],
                         ).to(kwargs['device'])

    t_dsnae = DSNBasisAE(shared_encoder=shared_encoder,
                         decoder=shared_decoder,
                         basis_vec=basis_vec,
                         input_dim=kwargs['input_dim'],
                         latent_dim=kwargs['latent_dim'],
                         testing_drug_len=kwargs['testing_drug_len'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         inv_temp=kwargs['inv_temp'],
                         dop=kwargs['dop'],
                         cosine_flag=cosine_flag,
                         cns_basis_label_loss=False,
                         num_geo_layer=kwargs['num_geo_layer'],
                         norm_flag=kwargs['norm_flag'],

                         ).to(kwargs['device'])

    confounding_classifier = MLP(input_dim=kwargs['latent_dim'] * 2,
                                 output_dim=1,
                                 hidden_dims=kwargs['classifier_hidden_dims'],
                                 dop=kwargs['dop']).to(kwargs['device'])


    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters(),
                 basis_vec,
                 ]
    t_ae_params = [t_dsnae.private_encoder.parameters(),
                   s_dsnae.private_encoder.parameters(),
                   shared_decoder.parameters(),
                   shared_encoder.parameters(),
                   basis_vec,
                   ]

    ae_params_list = list(chain(*ae_params))
    ae_params_list = [param.detach() if not param.is_leaf else param for param in ae_params_list]
    ae_optimizer = torch.optim.AdamW(ae_params_list, lr=kwargs['lr'])
    classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=kwargs['lr'])
    t_ae_params_list = list(chain(*t_ae_params))
    t_ae_params_list = [param.detach() if not param.is_leaf else param for param in t_ae_params_list]

    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)
    critic_train_history = defaultdict(list)
    gen_train_history = defaultdict(list)

    if kwargs['retrain_flag']:
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                dsnae_train_history = basis_dsn_ae_train_step(s_dsnae=s_dsnae,
                                                              t_dsnae=t_dsnae,
                                                              s_batch=s_batch,
                                                              t_batch=t_batch,

                                                              device=kwargs['device'],
                                                              optimizer=ae_optimizer,
                                                              history=dsnae_train_history)

            dsnae_val_history = eval_basis_dsnae_epoch(model=s_dsnae,
                                                       data_loader=s_test_dataloader,
                                                       device=kwargs['device'],

                                                       history=dsnae_val_history
                                                       )
            dsnae_val_history = eval_basis_dsnae_epoch(model=t_dsnae,
                                                       data_loader=t_test_dataloader,

                                                       device=kwargs['device'],
                                                       history=dsnae_val_history
                                                       )

            for k in dsnae_val_history:
                if k != 'best_index':
                    dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                    dsnae_val_history[k].pop()
            if kwargs['es_flag']:
                save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=20)
                if save_flag:
                    torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_s_dsnae.pt'))
                    torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt'))
                if stop_flag:
                    break
        if kwargs['es_flag']:
            s_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'a_s_dsnae.pt')))
            t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt')))

        for epoch in range(int(kwargs['train_num_epochs'])):
            print("WGAN")
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                critic_train_history = critic_dsn_train_step(critic=confounding_classifier,
                                                             s_dsnae=s_dsnae,
                                                             t_dsnae=t_dsnae,
                                                             s_batch=s_batch,
                                                             t_batch=t_batch,
                                                             device=kwargs['device'],
                                                             optimizer=classifier_optimizer,
                                                             history=critic_train_history,
                                                             gp=10.0)
                print(step)
                if (step + 1) % 2 == 0:
                    gen_train_history = gan_dsn_gen_train_step(critic=confounding_classifier,
                                                               s_dsnae=s_dsnae,
                                                               t_dsnae=t_dsnae,
                                                               s_batch=s_batch,
                                                               t_batch=t_batch,
                                                               device=kwargs['device'],
                                                               optimizer=t_ae_optimizer,
                                                               alpha=1.0,

                                                               history=gen_train_history)

        torch.save(s_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_s_dsnae.pt'))
        torch.save(t_dsnae.state_dict(), os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt'))

    else:
        try:
            t_dsnae.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'a_t_dsnae.pt')))

        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return t_dsnae.shared_encoder, (
        dsnae_train_history, dsnae_val_history, critic_train_history, gen_train_history), basis_vec, inv_temp