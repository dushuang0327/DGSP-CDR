import torch
from torch import nn
from torch.nn import functional as F
from src.base_ae import BaseAE
from src.types_ import *
from typing import List, Tuple


class DSNAE(BaseAE):

    def __init__(self, shared_encoder, decoder, input_dim: int, latent_dim: int, alpha: float = 1.0,
                 hidden_dims: List = None, dop: float = 0.1, noise_flag: bool = False, norm_flag: bool = False,
                 **kwargs) -> None:
        super(DSNAE, self).__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.noise_flag = noise_flag
        self.dop = dop
        self.norm_flag = norm_flag
        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.shared_encoder = shared_encoder
        self.decoder = decoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )
        modules.append(nn.Dropout(self.dop))
        modules.append(nn.Linear(hidden_dims[-1], latent_dim, bias=True))

        self.private_encoder = nn.Sequential(*modules)

        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def p_encode(self, input: Tensor) -> Tensor:
        if self.noise_flag and self.training:
            latent_code = self.private_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            latent_code = self.private_encoder(input)

        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def s_encode(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.noise_flag and self.training:
            latent_code = self.shared_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            latent_code = self.shared_encoder(input)
        mean = self.fc_mean(latent_code)
        logvar = self.fc_logvar(latent_code)
        if self.norm_flag:
            return mean, logvar, F.normalize(latent_code, p=2, dim=1)
        else:
            return mean, logvar, latent_code

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mean, logvar, s_latent_code = self.s_encode(input)
        p_latent_code = self.p_encode(input)
        z_reparam = self.reparameterize(mean, logvar)
        z = torch.cat((p_latent_code, s_latent_code), dim=1)
        return z_reparam, mean, logvar, z

    def decode(self, z_reparam: Tensor) -> Tensor:
        outputs = self.decoder(z_reparam)
        return outputs

    def forward(self, input: Tensor) -> List[Tensor]:
        z_reparam, mean, logvar, z = self.encode(input)
        return [input, self.decode(z_reparam), z_reparam, mean, logvar]

    def reparameterize(self, mean, logvar) -> Tensor:
        std = torch.exp(0.5 * logvar.clamp(min=-10, max=10))
        eps = torch.randn_like(std)
        return mean + eps * std

    def loss_function(self, *args, **kwargs) -> dict:
        input = args[0]
        recons = args[1]
        z_reparam = args[2]
        mean = args[3]
        logvar = args[4]
        z = args[5]

        p_z = z[:, :z.shape[1] // 2]
        s_z = z[:, z.shape[1] // 2:]

        recons_loss = F.mse_loss(input, recons)

        batch_size = input.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size

        s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)
        p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)
        ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))

        loss = recons_loss + kl_loss + self.alpha * ortho_loss
        return {'loss': loss, 'recons_loss': recons_loss, 'kl_loss': kl_loss, 'ortho_loss': ortho_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]
