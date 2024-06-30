import torch
import torch.nn as nn
import random

class SaveMuVar():
    mu, var = None, None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.mu = output.detach().cpu().mean(dim=[2, 3], keepdim=True).squeeze(-1).squeeze(-1).numpy()
        self.var = output.detach().cpu().var(dim=[2, 3], keepdim=True).squeeze(-1).squeeze(-1).numpy()

    def remove(self):
        self.hook.remove()

class EFDMix(nn.Module):
    """EFDMix.

    Reference:
      Zhang et al. Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization. CVPR 2022.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
        x_view = x.view(B, C, -1)
        value_x, index_x = torch.sort(x_view)  # sort inputs
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)
        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError

        inverse_index = index_x.argsort(-1)
        x_view_copy = value_x[perm].gather(-1, inverse_index)
        new_x = x_view + (x_view_copy - x_view.detach()) * (1-lmda)
        return new_x.view(B, C, W, H)

class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True  # Train: True, Test: False

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        if not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)
        return x_normed*sig_mix + mu_mix

class TriD(nn.Module):
    """TriD.
    Reference:
      Chen et al. Treasure in Distribution: A Domain Randomization based Multi-Source Domain Generalization for 2D Medical Image Segmentation. MICCAI 2023.
    """
    def __init__(self, p=0.5, eps=1e-6, alpha=0.1):
        """
        Args:
          p (float): probability of using TriD.
          eps (float): scaling parameter to avoid numerical issues.
          alpha (float): parameter of the Beta distribution.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        if random.random() > self.p:
            return x

        N, C, H, W = x.shape

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        # Sample mu and var from an uniform distribution, i.e., mu ～ U(0.0, 1.0), var ～ U(0.0, 1.0)
        mu_random = torch.empty((N, C, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).to(x.device)
        var_random = torch.empty((N, C, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).to(x.device)

        lmda = self.beta.sample((N, C, 1, 1))
        bernoulli = torch.bernoulli(lmda).to(x.device)

        mu_mix = mu_random * bernoulli + mu * (1. - bernoulli)
        sig_mix = var_random * bernoulli + sig * (1. - bernoulli)
        return x_normed * sig_mix + mu_mix

class MixNoise(nn.Module):
    def __init__(self, batch_size=4, p=0.5, mix_style=True, no_noise=False,
                 mix_learnable=True, noise_learnable=True, always_use_beta=False, alpha=0.1, eps=1e-6, use_gpu=True,
                 debug=False):
        super().__init__()
        self.batch_size = batch_size
        self.p = p
        self.mix_style = mix_style
        self.no_noise = no_noise
        self.mix_learnable = mix_learnable
        self.noise_learnable = noise_learnable
        self.always_use_beta = always_use_beta
        self.alpha = alpha
        self.eps = eps
        self.use_gpu = use_gpu
        self.debug = debug
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.data = None
        self.init_parameters()

    def init_parameters(self):
        batch_size = self.batch_size

        self.perm = torch.randperm(batch_size)
        while torch.allclose(self.perm, torch.arange(batch_size)):
            # avoid identical permutation order
            self.perm = torch.randperm(batch_size)

        if self.debug: print('permutation index', self.perm)

        self.rand_p = torch.rand(1)

        if self.rand_p >= self.p:
            ##not performing
            if self.debug: print("not performing maxstyle")
            self.lmda = torch.zeros(batch_size, 1, 1, 1, device=self.device).float()
            self.lmda.requires_grad = False
        else:
            if self.mix_style is False:
                self.lmda = torch.zeros(batch_size, 1, 1, 1, dtype=torch.float32, device=self.device)
                self.lmda.requires_grad = False
            else:
                self.lmda = None
                if self.always_use_beta:
                    self.beta_sampler = torch.distributions.Beta(self.alpha, self.alpha)
                    lmda = self.beta_sampler.sample((batch_size, 1, 1, 1)).to(self.device)
                    self.lmda = nn.Parameter(lmda.float())
                else:
                    lmda = torch.rand(batch_size, 1, 1, 1, dtype=torch.float32, device=self.device)
                    self.lmda = nn.Parameter(lmda.float())

                if self.mix_learnable:
                    self.lmda.requires_grad = True
                else:
                    self.lmda.requires_grad = False
        self.gamma_std = None
        self.beta_std = None

    def __repr__(self):
        if self.p >= self.rand_p:
            return f'MaxStyle: \
                 mean of gamma noise: {torch.mean(self.gamma_noise)}, std:{torch.std(self.gamma_noise)} ,\
                 mean of beta noise: {torch.mean(self.beta_noise)}, std: {torch.std(self.beta_noise)}, \
                 mean of mix coefficient: {torch.mean(self.lmda)}, std: {torch.std(self.lmda)}'

        else:
            return 'diffuse style not applied'

    def reset(self):
        ## this function must be called before each forward pass if the underlying input data is changed
        self.init_parameters()
        if self.debug: print('reinitializing parameters')

    def forward(self, x, gamma_noise, beta_noise):
        if random.random() > self.p:
            return x
        self.num_feature = x.size(1)
        self.data = x
        B = x.size(0)
        C = x.size(1)
        flatten_feature = x.view(B, C, -1)
        if (self.rand_p >= self.p) or (not self.mix_style and self.no_noise) or B <= 1 or flatten_feature.size(2) == 1:
            # print("not operating")
            if B <= 1:
                Warning('MaxStyle: B<=1, not performing maxstyle')
            if flatten_feature.size(2) == 1:
                Warning('MaxStyle: spatial dim=1, not performing maxstyle')
            return x

        assert self.batch_size == B and self.num_feature == C, f"check input dim, expect ({self.batch_size}, {self.num_feature}, *,*) , got {B},{C}"

        # style normalization, making it consistent with MixStyle implementation
        mu = x.mean(dim=[2, 3], keepdim=True)  ## [B,C,1,1] 均值
        var = x.var(dim=[2, 3], keepdim=True)  ## [B,C,1,1] 方差
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        gamma_std = torch.std(sig, dim=0, keepdim=True).detach()  ## [1,C,1,1]
        beta_std = torch.std(mu, dim=0, keepdim=True).detach()  ## [1,C,1,1]

        if self.mix_style:
            clipped_lmda = torch.clamp(self.lmda, 0, 1)
            mu2, sig2 = mu[self.perm], sig[self.perm]  ## [B,C,1,1]
            sig_mix = sig * (1 - clipped_lmda) + sig2 * clipped_lmda  ## [B,C,1,1]
            mu_mix = mu * (1 - clipped_lmda) + mu2 * clipped_lmda
        else:
            sig_mix = sig
            mu_mix = mu

        a = gamma_noise * gamma_std
        b = beta_noise * beta_std
        x_aug = (sig_mix + gamma_noise * gamma_std) * x_normed + (mu_mix + beta_noise * beta_std)

        return x_aug

class DSU(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-8, mix='gaussian', lmda=None, zero_init=False, coefficient_sampler=None):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          lmda (float): weight for mix (intrapolation or extrapolation). If not set, will sample from [0,1) (from beta distribution)
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self.mu = None
        self.std = None
        self.zero_init = zero_init
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.lmda = lmda
        self.coeficient_sampler = None
        self.beta = torch.distributions.Beta(alpha, alpha)

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def get_perm(self):
        return self.perm

    def forward(self, x, perm=None):
        p = torch.rand(1)
        if p > self.p:
            # print("not operating")
            return x

        B = x.size(0)
        C = x.size(1)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        if self.lmda is None:
            if self.coeficient_sampler is None:
                # use default setting: for intrapolatio, e.g., mixstyle,  use beta distribution and for extrapolation, gaussian uses uniform distribution
                # print(f"lamda: {lmda}")
                lmda = self.beta.sample((B, 1, 1, 1))
            else:
                if self.coeficient_sampler == 'beta':
                    lmda = self.beta.sample((B, 1, 1, 1))
                elif self.coeficient_sampler == 'uniform':
                    lmda = torch.rand(B, 1, 1, 1)
                elif self.coeficient_sampler == 'gaussian':
                    lmda = torch.randn(B, 1, 1, 1)
                else:
                    raise ValueError

        else:
            lmda = torch.ones(B, 1, 1, 1) * self.lmda
        lmda = lmda.to(x.device)

        if self.mix in ['random', 'crossdomain']:
            if perm is not None:
                mu2, sig2 = mu[perm], sig[perm]
            else:
                if self.mix == 'random':
                    # random shuffle
                    perm = torch.randperm(B)
                    mu2, sig2 = mu[perm], sig[perm]
                elif self.mix == 'crossdomain':
                    # split into two halves and swap the order
                    perm = torch.arange(B - 1, -1, -1)  # inverse index
                    perm_b, perm_a = perm.chunk(2)
                    perm_b = perm_b[torch.randperm(B // 2)]
                    perm_a = perm_a[torch.randperm(B // 2)]
                    perm = torch.cat([perm_b, perm_a], 0)
                    mu2, sig2 = mu[perm], sig[perm]
            mu_mix = mu * (1 - lmda) + mu2 * lmda
            sig_mix = sig * (1 - lmda) + sig2 * lmda
            self.perm = perm
            # print(perm)
            return x_normed * sig_mix + mu_mix
        elif self.mix == 'gaussian':
            ## DSU: adding gaussian noise
            gaussian_mu = torch.randn(B, C, 1, 1, device=x.device) * torch.std(mu, dim=0, keepdim=True)
            gaussian_mu.requires_grad = False
            gaussian_std = torch.randn(B, C, 1, 1, device=x.device) * torch.std(sig, dim=0, keepdim=True)
            gaussian_std.requires_grad = False
            mu_mix = mu + gaussian_mu
            sig_mix = sig + gaussian_std
            return x_normed * sig_mix + mu_mix
        else:
            raise NotImplementedError

