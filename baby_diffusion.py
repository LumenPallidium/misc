import torch

## see diffusion paper
## https://arxiv.org/pdf/2006.11239.pdf

def corrupt_tensor(x, alpha):
    """
    Generate a latent and corruption of the input,
    based on equation 14 of the diffusion paper.

    Parameters
    ----------
    x : torch.Tensor
        The tensor to interpolate.
    alpha : torch.Tensor
        Tensor of interpolation parameters, same
        shape as x. 0 = no noise, 1 = all noise.

    Returns
    -------
    z : torch.Tensor
        The latent variable
    x_noise : torch.Tensor
        The corrupted variable
    """
    z = torch.randn_like(x, device = x.device)
    sqrt_alpha = torch.sqrt(alpha)[:, None, None, None]
    sqrt_alpha_inv = torch.sqrt(1 - alpha)[:, None, None, None]
    return z, sqrt_alpha * x + sqrt_alpha_inv * z

def diffusion_loss(model, x, conditioning = None):
    """
    A single training step for diffusion, see Algorithm 1
    in the diffusion paper.

    Essentially, randomly choose a step in the diffusion process,
    corrupt the input to that step. Then, estimate the corruption
    from the corrupted input, and use get the MSE loss.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train. Assumed to have an alpha_bars
        attribute, which is a tensor of shape (diffusion_steps,).
    x : torch.Tensor
        The input tensor.

    Returns
    -------
    loss : torch.Tensor
        The loss for this step.
    """
    batch_size = x.shape[0]
    # generate steps randomly and get the alpha for that step
    step_tensor = torch.randint(0,
                                model.diffusion_steps,
                                (batch_size,),
                                device = model.device)
    alphas = model.alpha_bars.index_select(0, step_tensor)
    # corrup the batch
    z, x_noise = corrupt_tensor(x, alphas)
    # estimate corruption (embedding the steps)
    z_hat = model(x_noise, timesteps = step_tensor, conditioning = conditioning)
    # use MSE loss by default
    loss = torch.nn.functional.mse_loss(z_hat, z)
    return loss

def sample(model, n_samples = 1, conditioning = None):
    """
    Sample from the model. See algorithm 2 in the diffusion paper.
    Essentially, generate a batch of noise, and then run the model
    backwards in time for each step to get the samples.

    Parameters
    ----------
    model : torch.nn.Module
        The model to sample from. Assumed to have a
        device and in_shape attribute.
    n_samples : int
        The number of samples to generate.

    Returns
    -------
    x : torch.Tensor
        The sampled tensor (an image here).
    """
    with torch.no_grad():
        image = torch.randn(n_samples, *model.in_shape, device = model.device)

        for t in reversed(range(model.diffusion_steps)):
            if t == 0:
                noise = torch.zeros(n_samples, *model.in_shape, device = model.device)
            else:
                noise = torch.randn(n_samples, *model.in_shape, device = model.device)
            # the diffusion parameters for this step
            alpha = model.alphas[t]
            alpha_bar = model.alpha_bars[t]
            sigma = model.sigmas[t]

            # make timestep a tensor for embedding
            t_tensor = torch.full((n_samples,), t, device = model.device)

            denoise = model(image, timesteps = t_tensor, conditioning = conditioning)
            # weird huh? see the paper
            denoise = (1 - alpha) * denoise / torch.sqrt(1 - alpha_bar)
            # we're subtracting tiny bits of noise at each step and scaling to avoid overgrowth
            image = (1 / torch.sqrt(alpha)) * (image - denoise)

            # lastly, add noise for the inverse diffusion
            image = image + sigma * noise
    return image

def embed_timesteps(timesteps, n_channels, multiplier = 10000):
    """
    Embed the timesteps using sine and cosine functions,
    as in the transformer paper. Yes, it's confusing.
    This is mentioned in Appendix B of the diffusion paper.

    What this does is gives each timestep a unique embedding,
    so that we can feed time information into the model.

    Also can be used for class conditioning.

    Parameters
    ----------
    timesteps : torch.Tensor
        The timesteps to embed.
    n_channels : int
        The number of channels to embed.

    Returns
    -------
    embedded : torch.Tensor
        The embedded timesteps. Shape is (timesteps, n_channels).
    """
    dim_steps = torch.arange(n_channels // 2,
                             dtype = torch.float32,
                             device = timesteps.device)
    dim_steps = multiplier ** (2 * dim_steps / n_channels)
    dim_steps = timesteps.float()[:, None] / dim_steps[None, :]
    embed = torch.cat([torch.sin(dim_steps), torch.cos(dim_steps)], dim = -1)
    return embed

class ResBlock(torch.nn.Module):
    """
    A classic resnet block, with layer norm instead of batch norm.

    Parameters
    ----------
    channels : int
        Number of input channels.
    kernel_size : int
        Kernel size for the convolutional layers.
    stride : int
        Stride for the convolutional layers.
    padding : int
        Padding for the convolutional layers.
    activation : torch.nn.Module
        Activation function to use.
    """
    def __init__(self,
                 channels,
                 in_channels = None,
                 kernel_size = 3,
                 stride = 1,
                 activation = torch.nn.GELU(),
                 embed_channels = 16):
        super().__init__()
        self.activation = activation
        # make an input convolution if needed
        if in_channels is None:
            self.in_conv = torch.nn.Identity()
        else:
            self.in_conv = torch.nn.Conv2d(in_channels, channels,
                                           kernel_size = 1,
                                           stride = stride,
                                           padding = "same")
            
        self.first_conv = torch.nn.Conv2d(channels, channels,
                                          kernel_size = kernel_size,
                                          stride = stride,
                                          padding = "same")
        # classic resnet uses batch norm but not us
        self.first_norm = torch.nn.LayerNorm(channels)
        self.second_conv = torch.nn.Conv2d(channels, channels,
                                           kernel_size = kernel_size,
                                           stride = stride,
                                           padding = "same")
        self.second_norm = torch.nn.LayerNorm(channels)

        self.time_embed = torch.nn.Linear(embed_channels, channels)

    def forward(self, x, embed = None):
        x = self.in_conv(x)
        res = self.first_conv(x)
        # need to permute to use layer norm
        # (batch, channels, height, width) -> (batch, height, width, channels)
        res = self.first_norm(res.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        res = self.activation(res)

        if embed is not None:
            embed = self.time_embed(embed)
            res = res + embed[:, :, None, None]

        res = self.second_conv(res)
        res = self.second_norm(res.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        res = self.activation(res)

        return x + res

class DiffusionModel(torch.nn.Module):
    """
    A U-Net style model for diffusion. Note that it contains
    the diffusion parameters as buffers.

    Parameters
    ----------
    in_shape : tuple
        The shape of the input tensor.
    timestep_embed_dim : int
        The number of channels to use for embedding timesteps.
    diffusion_steps : int
        The number of diffusion steps to use.
    channels : list
        The number of channels for the first layer in the U-Net.
    n_downsamples : int
        The number of downsamples to use in the U-Net.
    down_scale : int
        The scale factor for downsampling. Also the channel
        multiple to use for channel numbers in lower layers.
    kernel_size : int
        The kernel size for the convolutional layers.
    beta_min : float
        The minimum beta (noise variance) to use for diffusion.
    beta_max : float
        The maximum beta (noise variance) to use for diffusion.
    device : str
        The device to use for the model.
    """
    def __init__(self,
                 in_shape,
                 timestep_embed_dim = 16,
                 diffusion_steps = 1000,
                 channels = [8, 16, 32],
                 n_downsamples = 2,
                 down_scale = 2,
                 kernel_size = 3, 
                 beta_min = 1e-4,
                 beta_max = 2e-2,
                 device = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.in_shape = in_shape
        self.timestep_embed_dim = timestep_embed_dim
        self.device = device
        self.diffusion_steps = diffusion_steps
        self.n_downsamples = n_downsamples
        self.down_scale = down_scale


        self.diffusion_steps = diffusion_steps
        # from diffusion paper, section 4
        betas = torch.linspace(beta_min,
                               beta_max,
                               diffusion_steps)
        # second choice, see section 3.2
        sigmas = torch.sqrt(betas)
        # section 2, right above equation 4
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim = 0)

        # register as buffers so they're saved with the model
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("betas", betas)

        layers = []
        up_convs = []
        down_convs = []
        in_channels = in_shape[0]
        n_channels = in_channels

        for i in range(n_downsamples + 1):
            channel_scale = down_scale ** i
            layers_i = []
            
            for j in range(len(channels)):
                layers_i.append(ResBlock(channels[j] * channel_scale,
                                         in_channels = n_channels * channel_scale,
                                         kernel_size = kernel_size,
                                         embed_channels = timestep_embed_dim))
                n_channels = channels[j]

            down_convs.append(torch.nn.Conv2d(n_channels * channel_scale,
                                              n_channels * channel_scale * down_scale,
                                              kernel_size = down_scale,
                                              stride = down_scale))
            
            # we concat each upsample, so channels double
            mult = 1 if i == n_downsamples else 2
            out_channels = n_channels * channel_scale // down_scale if i > 0 else in_channels
            up_convs.append(torch.nn.Conv2d(n_channels * channel_scale * mult,
                                            out_channels,
                                            kernel_size = kernel_size,
                                            padding = "same"))

            layers.append(torch.nn.ModuleList(layers_i))
        
        self.layers = torch.nn.ModuleList(layers)
        self.up_convs = torch.nn.ModuleList(up_convs)
        self.down_convs = torch.nn.ModuleList(down_convs)

    def forward(self, x, timesteps = None, conditioning = None):
        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        embed : torch.Tensor
            The timesteps to embed
        conditioning : torch.Tensor
            The conditioning tensor.

        Returns
        -------
        x : torch.Tensor
            Output tensor, should essentially be the predicted noise.
        """
        if conditioning is not None:
            embed = embed_timesteps(timesteps, self.timestep_embed_dim // 2)
            # embed conditioning at a relatively prime frequency
            embed_cond = embed_timesteps(conditioning,
                                         self.timestep_embed_dim // 2,
                                         multiplier = 7 * 7 * 7)
            embed = torch.cat([embed, embed_cond], dim = -1)
        else:
            embed = embed_timesteps(timesteps, self.timestep_embed_dim)

        x_downs = []
        # first pass to get the downsampling part of unet
        for i in range(self.n_downsamples + 1):
            if x_downs:
                x_i = x_downs[-1]
                x_i = self.down_convs[i - 1](x_i)
            else:
                x_i = x.clone()

            # run model
            layer = self.layers[i]
            for sublayer in layer:
                x_i = sublayer(x_i, embed = embed)

            x_downs.append(x_i)

        x_ups = []
        # next pass combines and upsamples
        for i in range(self.n_downsamples, -1, -1):
            x_i = x_downs[i]

            # concat with next layer channel-wise
            if x_ups:
                x_i = torch.cat([x_i, x_ups[-1]], dim = 1)

            if i != 0:
                # upsample one step
                x_i = torch.nn.functional.interpolate(x_i,
                                                    scale_factor = self.down_scale,
                                                    mode = "bilinear")

            x_i = self.up_convs[i](x_i)
            x_ups.append(x_i)

        return x_i

if __name__ == "__main__":
    import tqdm
    import os
    import numpy as np
    from copy import deepcopy
    from PIL import Image
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    os.makedirs("samples", exist_ok = True)

    n_epochs = 100
    batch_size = 1024
    ema_decay = 0.99
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = CIFAR10("data", download = True, transform = transform)

    model = DiffusionModel((3, 32, 32), device = device)
    model_ema = deepcopy(model)
    model.to(device)
    model_ema.to(device)

    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    pbar = tqdm.trange(n_epochs * len(cifar) // batch_size)
    for epoch in range(n_epochs):
        dataloader = torch.utils.data.DataLoader(cifar,
                                                 batch_size = batch_size,
                                                 shuffle = True,)
        for (x, y) in dataloader:
            opt.zero_grad()
            x, y = x.to(device), y.to(device)
            loss = diffusion_loss(model, x, conditioning = y)
            loss.backward()
            opt.step()

            pbar.set_description(f"loss : {loss.item():.4f}")

            if (pbar.n % 500) == 0:
                condition = torch.arange(10, device = device)
                samples = sample(model_ema, n_samples = 10, conditioning = condition)
                samples = samples.detach().cpu()
                samples = samples.permute(0, 2, 3, 1).squeeze()
                samples = np.hstack(samples.numpy())
                samples = (samples * 255).astype(np.uint8)
                step = str(pbar.n).zfill(5)
                Image.fromarray(samples).save(f"samples/diffusion_{step}.png")
            pbar.update(1)

            # update the ema model
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.data.mul_(ema_decay)
                param_ema.data.add_((1 - ema_decay) * param.data)



