"""
Arquitectura del VAE de SELFIES + cabeza de propiedad.
========================================================
Encoder y decoder se preentrenan sobre ZINC250k y luego se CONGELAN.
Solo la PropertyHead se entrena despues, con los 18 puntos de E_ads.

Dimensiones de arranque (documentadas, no magicas):
  - embedding_dim = 64
  - hidden_size   = 256   (encoder y decoder)
  - latent_dim    = 56    (valor usado en Gomez-Bombarelli et al., 2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=256, latent_dim=56, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_size * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len) indices de tokens
        emb = self.embed(x)
        _, h = self.gru(emb)  # h: (2, batch, hidden_size) por ser bidireccional
        h = torch.cat([h[0], h[1]], dim=-1)  # (batch, hidden_size*2)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=256, latent_dim=56, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, z, target_seq):
        """Teacher forcing: decodifica target_seq desplazado, condicionado en z."""
        h0 = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)  # (1, batch, hidden_size)
        emb = self.embed(target_seq)
        out, _ = self.gru(emb, h0)
        logits = self.out(out)  # (batch, seq_len, vocab_size)
        return logits


class PropertyHead(nn.Module):
    """La UNICA parte que se entrena con los 18 puntos etiquetados."""
    def __init__(self, latent_dim=56, hidden=(32, 16)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class SelfiesVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=256, latent_dim=56, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_size, latent_dim, pad_idx)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size, latent_dim, pad_idx)
        self.property_head = PropertyHead(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, x[:, :-1])  # predice el siguiente token
        return logits, mu, logvar, z

    def freeze_encoder_decoder(self):
        """Llamar ANTES de la fase de fine-tuning con los 18 datos."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False


def vae_loss(logits, target, mu, logvar, pad_idx=0, beta=1.0):
    """Perdida de preentrenamiento: reconstruccion (cross-entropy) + beta * KL.
    beta debe subir gradualmente durante entrenamiento (KL annealing) -
    si beta=1 desde el inicio, el modelo colapsa el espacio latente (posterior collapse)."""
    target_shifted = target[:, 1:]  # el token que el decoder debe predecir en cada paso
    recon_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target_shifted.reshape(-1),
        ignore_index=pad_idx, reduction="mean",
    )
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


def kl_anneal_schedule(step, total_steps, warmup_frac=0.3, max_beta=0.5):
    """Beta sube linealmente de 0 a max_beta durante el primer warmup_frac del entrenamiento."""
    warmup_steps = warmup_frac * total_steps
    return min(max_beta, max_beta * step / max(1, warmup_steps))
