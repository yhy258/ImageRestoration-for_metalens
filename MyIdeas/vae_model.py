import math

import torch
from torch.nn import functional as F
import torch.distributions as dist

from torch import nn
from abc import abstractmethod

# from utils import kl_divergence
# from utils import normal_kl_divergence

from typing import List, Callable, Union, Any, TypeVar, Tuple


Tensor = TypeVar('torch.tensor')


# Prior 고려할 수 있게끔 prior 정보 부여하는 encoder decoder ㅇㅇ
# 메타렌즈의 경우 여러 손상 요소들이 섞여 있음 (맞는지 확인)
# Conditional Latent Vector를 MultiHead로 부여 (개선된 아키텍쳐 )

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 head_nums: int, # 1 이상
                 hidden_dims: List = None,
                 image_size=(256, 256),
                 device='cuda' if torch.cuda.is_available() else "cpu",
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()


        self.latent_dim = latent_dim
        out_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]

        # Build Encoder
        for i, h_dim in enumerate(hidden_dims):
            if i < 2 :
                strides = 2
            else :
                strides = 1
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= strides, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.conv_mu = nn.Conv2d(hidden_dims[-1], latent_dim * head_nums, kernel_size=1, groups=head_nums)
        self.conv_var = nn.Conv2d(hidden_dims[-1], latent_dim * head_nums, kernel_size=1, groups=head_nums)
        # 이런식으로 했다가 잘 안되면 GAP 사용 고려.

        # Build Decoder
        modules = []

        self.decoder_input = nn.Conv2d(latent_dim * head_nums, hidden_dims[-1], kernel_size=1)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            if i < 2 :
                strides = 2
                output_padding = 1
            else :
                strides = 1
                output_padding = 0
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = strides,
                                       padding=1,
                                       output_padding=output_padding),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels = out_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

        # H, W = int(image_size[0] / 4.), int(image_size[1] / 4.)
        # self.p_dist = dist.normal.Normal(loc=torch.zeros( latent_dim*head_nums*H*W ).to('cuda'), scale=torch.ones( latent_dim*head_nums*H*W ).to('cuda'))

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.conv_mu(result) # 2d 형태의 ...
        log_var = self.conv_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x C' x H' x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = args[4]

        recons_loss = F.l1_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def d_loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        beta = args[4]
        alpha = args[5]
        # regs = args[6]

        bs = recons.shape[0]

        recons_loss = F.l1_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1))

        # analytic하게 수정.
        # q(z) = E_{x~q(x))}[q(z|x)]
        mu_ = torch.mean(mu, dim=0, keepdim=True)
        var = log_var.exp()
        var_ = torch.mean(var, dim=0, keepdim=True)
        structure_loss = torch.mean(-0.5 * torch.mean(1 + torch.log(var_) - mu_ ** 2 - var_, dim=1))


        loss = recons_loss + beta * kld_loss + alpha * structure_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD':-kld_loss.detach(), "Structure Divergence":structure_loss.detach()}


    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


from distributions import PowerSpherical, HypersphericalUniform, _kl_powerspherical_uniform
class PowerVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 head_nums: int, # 1 이상
                 hidden_dims: List = None,
                 image_size: int = (256, 256),
                 **kwargs) -> None:
        super(PowerVAE, self).__init__()


        self.latent_dim = latent_dim
        self.head_nums = head_nums
        self.image_size = image_size
        H, W = image_size[0]/4., image_size[1]/4.
        out_channels = in_channels

        self.q_dist = PowerSpherical

        self.p_dist = HypersphericalUniform(dim=latent_dim*head_nums*H*W, device="cuda")

        self.mu = torch.zeros(1, latent_dim*head_nums, H, W).to("cuda")

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]

        # Build Encoder
        for i, h_dim in enumerate(hidden_dims):
            if i < 2 :
                strides = 2
            else :
                strides = 1
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= strides, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.conv_scale = nn.Conv2d(hidden_dims[-1], latent_dim * head_nums, kernel_size=1, groups=head_nums)
        # 이런식으로 했다가 잘 안되면 GAP 사용 고려.

        # Build Decoder
        modules = []

        self.decoder_input = nn.Conv2d(latent_dim * head_nums, hidden_dims[-1], kernel_size=1)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            if i < 2 :
                strides = 2
                output_padding = 1
            else :
                strides = 1
                output_padding = 0
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = strides,
                                       padding=1,
                                       output_padding=output_padding),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels = out_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu
        scale = self.conv_scale(result)

        return [mu, scale]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x C' x H' x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, scale: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        q_zx = self.q_dist(loc=self.mu.view(self.mu.shape[0], -1), scale=scale.view(self.mu.shape[0], -1))
        H, W = self.image_size[0] / 4, self.image_size[1] / 4,
        return q_zx.rsample(self.mu.shape[0]).view(self.mu.shape[0], self.latent_dim*self.head_nums, H, W)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, scale = self.encode(input)

        q_zx = self.q_dist(loc=mu.view(mu.shape[0], -1), scale=scale.view(mu.shape[0], -1))
        z = q_zx.rsample(mu.shape[0])
        H = W = self.image_size/ 4
        z = z.view(mu.shape[0], self.latent_dim*self.head_nums, H, W)
        return  [self.decode(z), input, mu, scale]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_scale = args[3]
        kld_weight = args[4]

        scale = log_scale.exp()

        dim = scale.shape[1] * scale.shape[2] * scale.shape[3]

        a = (dim - 1) / 2 + scale
        b = (dim - 1) / 2

        recons_loss = F.l1_loss(recons, input)

        kld_loss = power_entropy(scale, a, b).mean()


        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def d_loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_scale = args[3]
        beta = args[4]
        alpha = args[5]
        regs = args[6]

        # mu에 의존적이지 않다. 오직 scale만. mu를 constant처럼 바라봐도 좋을 것 같다.

        bs = recons.shape[0]

        recons_loss = F.l1_loss(recons, input)
        scale = log_scale.exp()


        dim = scale.shape[1] * scale.shape[2] * scale.shape[3]
        a = (dim - 1)/2 + scale
        b = (dim - 1)/2


        kld_loss = power_entropy(scale, a, b).mean()

        structure_loss = power_entropy(torch.mean(scale, dim=0, keepdim=True), a, b).mean()

        loss = recons_loss + beta * kld_loss + alpha * structure_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD':-kld_loss.detach(), "Structure Divergence":structure_loss.detach()}


    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = self.p_dist.sample(num_samples)

        z = z.to(current_device)

        H, W = self.image_size[0] / 4, self.image_size[1] / 4

        z = z.view(num_samples, self.latent_dim*self.head_nums, H, W)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


_EPS = 1e-7
def log_normalizer(alpha, beta):
    return -(
            (alpha + beta) * math.log(2)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + beta)
            + beta * math.log(math.pi)
    )

def power_entropy(scale, alpha, beta):
    return -(
            log_normalizer(alpha, beta)
            + scale
            * (math.log(2) + torch.digamma(alpha) - torch.digamma(alpha + beta))
    )