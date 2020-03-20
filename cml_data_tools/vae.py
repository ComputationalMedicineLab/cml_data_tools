"""VAE models and losses"""
import math
import torch
from torch import tensor
from torch.nn import BCELoss, Linear, Module
from torch.nn import functional as F
from torch.distributions import Bernoulli, Normal


class VAE(Module):
    """Variational AutoEncoder Torch Module.

    Parameters
    ----------
    encoder : torch.nn.Module
    decoder : torch.nn.Module
        pytorch networks used to encode and decode to/from the latent variables
    e : int
        The output dimension of the encoder network
    z : int
        The input dimension of the decoder network
    """
    def __init__(self, encoder, decoder, e, z):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mu = Linear(e, z)
        self.logvar = Linear(e, z)
        self.e = e
        self.z = z

    def encode(self, x):
        """Produce a mean and log variance from input"""
        h = F.relu(self.encoder(x))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        """Produce a sample from mu & log var that is differentiable"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        """Returns predictions and the generating distribution params"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


class VAELoss(Module):
    """Basic Loss function supporting VAE and Beta-VAE Modules

    Parameters
    ----------
    recon_fn : Callable
        A module instance or function to compute the reconstruction loss.
        Defaults to BCELoss(reduction='sum') (i.e., implicitly interprets the
        VAE output as Bernoulli)
    beta : int
        A beta term to apply to the KL Divergence term of the loss. Default=1
    """
    def __init__(self, recon_fn=BCELoss(reduction='sum'), beta=1):
        super().__init__()
        self.recon_fn = recon_fn
        self.beta = beta

    def forward(self, y_pred, y_true):
        # Match the criterion API expected by skorch.NeuralNet
        x_hat, mu, logvar = y_pred
        rec = self.recon_fn(x_hat, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return rec + self.beta*kld


class BetaTCVAE(Module):
    """beta-TC class VAE

    Parameters
    ----------
    encoder : torch.nn.Module
    decoder : torch.nn.Module
        pytorch networks used to encode and decode to/from the latent variables
    e : int
        The output dimension of the encoder network
    z : int
        The input dimension of the decoder network
    beta : int
        Total Correlation weight term (default=1)
    lamb : float in [0, 1]
        Dimension wise KL term is (1 - lamb)
    """
    def __init__(self, encoder, decoder, e, z, beta=1, lamb=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.e_dim = e
        self.z_dim = z
        self.beta = beta
        self.lamb = lamb
        # Learned Z Hyperparams
        # Q: Why logvar and not stddev?
        # A: https://stats.stackexchange.com/a/353222
        self.mu = Linear(e, z)
        self.logvar = Linear(e, z)

    def encode(self, x):
        """Hook for reparameterizing the outs of the encoder"""
        h = self.encoder(x)
        mu = self.mu(h)
        std = torch.exp(0.5*self.logvar(h))
        eps = torch.randn_like(std)
        z = mu + std*eps
        return z, mu, std

    def get_xdist(self, z):
        """Hook for customising interpretation of decoder output"""
        return Bernoulli(logits=self.decoder(z))

    def get_pdist(self, z):
        """Hook to customize prior distribution"""
        return Normal(torch.zeros_like(z), torch.ones_like(z))

    def get_qdist(self, mu, std):
        """Hook to customize construction of qdist from mean and stddev"""
        return Normal(mu, std)

    def forward(self, x, dataset_size):
        """Calculates the Evidence Lower Bound (ELBO) of the VAE on x"""
        x_len = x.shape[0]
        z, mu, std = self.encode(x)

        # log(p(x))
        xdist = self.get_xdist(z)
        logpx = xdist.log_prob(x).view(x_len, -1).sum(1)

        # log(p(z))
        pdist = self.get_pdist(z)
        logpz = pdist.log_prob(z).view(x_len, -1).sum(1)

        # log(q(z|x))
        qdist = self.get_qdist(mu, std)
        logqz_condx = qdist.log_prob(z).view(x_len, -1).sum(1)

        # Calculate matrix of shape (x_len, x_len, z_dim) which contains the
        # log probability of each instance's latent variables under the
        # distributions of every instance latent vars in the batch
        qdist = qdist.expand((1, x_len, self.z_dim))
        qzmat = qdist.log_prob(z.view(x_len, 1, self.z_dim))

        # log(q(z)) via minibatch weighted sampling
        logmn = math.log(dataset_size * x_len)
        logqz = torch.logsumexp(qzmat.sum(2), dim=1) - logmn
        logqz_prodmarginals = (torch.logsumexp(qzmat, dim=1) - logmn).sum(1)

        # Calculate Modified ELBO:
        # Basic ELBO is just logpx + logpz - logqz_condx
        ix_code_mi = logqz_condx - logqz
        total_corr = self.beta * (logqz - logqz_prodmarginals)
        dimwise_kl = (1 - self.lamb) * (logqz_prodmarginals - logpz)
        modified_elbo = logpx - ix_code_mi - total_corr - dimwise_kl
        return modified_elbo


class BetaTCVAELoss(Module):
    def forward(self, y_pred, y_true):
        # Match skorch.NeuralNet's expected criterion
        # y_pred is ELBO if used with BetaTCVAE
        return -1 * y_pred.mean()
