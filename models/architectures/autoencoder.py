import torch
import torch.nn as nn

from configs.config1 import INPUT_VARS, LATENT_DIM, ENCODER_HIDDEN, DECODER_HIDDEN


## Encoder ##
## Per-depth MLP -> masked mean-pool -> latent profile vector p
## Input:  profile (batch, depth, n_vars), mask (batch, depth, n_vars)
## Output: p (batch, latent_dim)

class Encoder(nn.Module):

    def __init__(self, n_vars=None, latent_dim=LATENT_DIM, hidden=ENCODER_HIDDEN):
        super().__init__()
        n_vars = n_vars or len(INPUT_VARS)

        layers = []
        in_dim = n_vars
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, latent_dim)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, profile, mask):
        """
        profile : (batch, depth, n_vars)
        mask    : (batch, depth, n_vars) bool — True where data is real
        Returns : p (batch, latent_dim)
        """
        # embed each depth level independently
        h = self.mlp(profile)                        # (batch, depth, latent_dim)

        # use any-variable mask to weight depth levels
        depth_mask = mask.any(dim=-1, keepdim=True)  # (batch, depth, 1)
        depth_mask = depth_mask.float()

        # masked mean-pool over depth
        p = (h * depth_mask).sum(dim=1) / depth_mask.sum(dim=1).clamp(min=1)

        return p                                     # (batch, latent_dim)


## Decoder ##
## Expand p to each depth level, reconstruct INPUT_VARS
## Input:  p (batch, latent_dim), depth (int)
## Output: reconstruction (batch, depth, n_vars)

class Decoder(nn.Module):

    def __init__(self, n_vars=None, latent_dim=LATENT_DIM, hidden=DECODER_HIDDEN):
        super().__init__()
        n_vars = n_vars or len(INPUT_VARS)

        layers = []
        in_dim = latent_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, n_vars)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, p, depth):
        """
        p     : (batch, latent_dim)
        depth : int — number of depth levels to reconstruct
        Returns : (batch, depth, n_vars)
        """
        p_expanded = p.unsqueeze(1).expand(-1, depth, -1)  # (batch, depth, latent_dim)
        return self.mlp(p_expanded)                         # (batch, depth, n_vars)


## Autoencoder ##
## Wraps encoder + decoder, handles save/load

class Autoencoder(nn.Module):

    def __init__(self, n_vars=None, latent_dim=LATENT_DIM,
                 encoder_hidden=ENCODER_HIDDEN, decoder_hidden=DECODER_HIDDEN):
        super().__init__()
        n_vars = n_vars or len(INPUT_VARS)
        self.encoder = Encoder(n_vars, latent_dim, encoder_hidden)
        self.decoder = Decoder(n_vars, latent_dim, decoder_hidden)

    def forward(self, profile, mask):
        """
        profile : (batch, depth, n_vars)
        mask    : (batch, depth, n_vars)
        Returns : reconstruction (batch, depth, n_vars), p (batch, latent_dim)
        """
        p    = self.encoder(profile, mask)
        recon = self.decoder(p, depth=profile.shape[1])
        return recon, p

    def save(self, path, stats=None):
        """
        Save model weights and optionally normalization stats.
        stats : dict of {var: (mean, std)} from ArgoProfileDataset
        """
        torch.save({"model_state": self.state_dict(), "stats": stats}, path)
        print(f"Saved autoencoder to {path}")

    @classmethod
    def load(cls, path, device="cpu", **kwargs):
        """Load autoencoder from checkpoint, return (model, stats)."""
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(**kwargs)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        return model, ckpt.get("stats")