# noise_vae.py
import torch
import torch.nn as nn

class NoiseVAE(nn.Module):
    """
    1D Conv VAE for [B, 1, T].
    Deeper encoder/decoder (×6 downsample/upsample) for larger receptive field.
    NOTE: Train with T_model = ceil(T/64)*64. We'll pad inputs to T_model and crop back.
    """
    def __init__(self, T: int, z_dim: int = 256):
        super().__init__()
        self.T = int(T)           # T_model used when training the checkpoint
        self.z_dim = int(z_dim)

        def enc_blk(c_in, c_out):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=9, stride=2, padding=4),
                nn.InstanceNorm1d(c_out, affine=True),
                nn.SiLU(inplace=True),
            )

        self.enc = nn.Sequential(
            enc_blk(1,   64),   # T/2
            enc_blk(64,  128),  # T/4
            enc_blk(128, 256),  # T/8
            enc_blk(256, 256),  # T/16
            enc_blk(256, 256),  # T/32
            enc_blk(256, 256),  # T/64
        )

        self._feat_T = self.T // 64
        self._c_bot  = 256
        self._flat   = self._c_bot * self._feat_T

        self.fc_mu     = nn.Linear(self._flat, self.z_dim)
        self.fc_logvar = nn.Linear(self._flat, self.z_dim)

        self.fc_dec = nn.Linear(self.z_dim, self._flat)

        def dec_blk(c_in, c_out):
            return nn.Sequential(
                nn.ConvTranspose1d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.SiLU(inplace=True),
            )

        self.dec = nn.Sequential(
            dec_blk(256, 256),  # T/32
            dec_blk(256, 256),  # T/16
            dec_blk(256, 256),  # T/8
            dec_blk(256, 128),  # T/4
            dec_blk(128,  64),  # T/2
            dec_blk(64,   32),  # T
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)                   # [B,256,T/64]
        h = h.flatten(1)                  # [B, 256*(T/64)]
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)                                # [B, 256*(T/64)]
        h = h.view(-1, self._c_bot, self._feat_T)         # [B,256,T/64]
        xhat = self.dec(h)                                # [B,1,T]
        return xhat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

    @torch.no_grad()
    def recon(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu)
