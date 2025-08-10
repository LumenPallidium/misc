import torch
from tqdm import tqdm

def rk4_integrate(system, x0, dt, n_steps):
    """
    4th-order Runge–Kutta time integration.
    """
    state = x0
    for _ in range(n_steps):
        k1 = system.step(state)
        k2 = system.step(state + 0.5 * dt * k1)
        k3 = system.step(state + 0.5 * dt * k2)
        k4 = system.step(state + dt * k3)

        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state

def rand_partition(n_pieces, total,
                   min_mult = None,
                   smoothing = 0.5):
    """
    Probably inefficient function to generate partitions with n_pieces that sum to total.

    Has no elements equal to 0.
    """
    if min_mult is not None:
        assert total % min_mult == 0, "Total must be divisible by min_mult"
        total = total // min_mult
    partition = torch.rand(n_pieces - 1)
    # smoothing a bit - avoids tiny partitions
    partition = partition + smoothing
    partition = partition / partition.sum()
    partition = partition * (total - 1)
    partition = partition.ceil().to(torch.int)
    if partition.sum() < total:
        partition = torch.cat([partition,
                            torch.tensor([total - partition.sum()])])
    else:
        excess = partition.sum() - total
        partition = torch.cat([partition,
                               torch.tensor([1])])
        reverse_cumsum = (partition-1).flip(0).cumsum(0).flip(0)
        mask = reverse_cumsum <= excess
        partition[mask] = 1
        remainder = partition.sum() - total
        if remainder > 0:
            before_mask = mask.int().argmax() - 1
            partition[before_mask] -= remainder
    if min_mult is not None:
        partition = partition * min_mult
    return partition

def multimodal_gaussian_sample(n,
                               means = torch.linspace(-1, 1, 8),
                               stds = 0.01,
                               weights = None):
    if weights is None:
        weights = torch.ones(len(means)) / len(means)
    if isinstance(stds, float):
        stds = torch.ones(len(means)) * stds
    indices = torch.multinomial(weights, n, replacement=True)
    samples = torch.randn(n) * stds[indices] + means[indices]
    return samples
                               
class SynergisticKuramoto(torch.nn.Module):
    def __init__(self,
                 N,
                 K = 1.2,
                 connection_prob = 0.1,
                 noise_scale = 0.00,
                 block_indices = None):
        super().__init__()
        self.N = N
        self.K = K
        self.noise_scale = noise_scale
        
        omega = multimodal_gaussian_sample(N)
        self.register_buffer("omega", omega)

        if block_indices is None:
            A = (torch.rand(N, N) < connection_prob).float()
        else:
            # dense connections within blocks
            ones = [torch.ones(idx, idx) for idx in block_indices]
            A = torch.block_diag(*ones)
            A = A + (torch.rand(N, N) < connection_prob).float()

        A.diagonal().fill_(0)
        self.register_buffer("A", A)

    def forward(self, x, n_steps = 5, dt = 0.005):
        x = x % (2 * torch.pi)
        x = rk4_integrate(self, x, dt, n_steps)
        x = x % (2 * torch.pi)
        return x

    def step(self, x):
        with torch.no_grad():
            sin_term = torch.sin(x[:, :, None] - x[:, None, :])
            connection = torch.sum(self.A[None, :, :] * sin_term, dim = 2)
            dx = self.omega[None, :] + self.K * connection / self.N
            if self.noise_scale > 0:
                noise = torch.randn_like(dx) * self.noise_scale
                dx += noise
        return dx
    
    def get_phase(self, x):
        x_complex = x.to(torch.complex64)
        rpsi = torch.exp(1j * x_complex) / self.N
        r = torch.abs(rpsi)
        psi = torch.angle(rpsi)
        return r, psi


class Lorenz96(torch.nn.Module):
    def __init__(self, N, F = 8.0):
        super().__init__()
        self.N = N
        self.F = F

    def forward(self, x, n_steps = 5, dt = 0.005):
        x = rk4_integrate(self, x, dt, n_steps)
        return x

    def step(self, x):
        with torch.no_grad():
            dx = torch.zeros_like(x)
            dx = (x.roll(-1, dims=1) - x.roll(2, dims=1)) * x.roll(1, dims=1) - x + self.F
        return dx

class MLP(torch.nn.Module):
    def __init__(self, 
                 dim, 
                 hidden_dim,
                 out_dim = None,
                 dropout = 0.,
                 activation = torch.nn.GELU,
                 residual = True):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        else:
            residual = False
        self.dim = dim
        self.out_dim = out_dim
        self.residual = residual
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.residual:
            return x + self.net(x)
        return self.net(x)
    
class Attention(torch.nn.Module):
    def __init__(self, 
                 dim,
                 n_heads = 8,
                 dropout = 0.,
                 bias = False,
                 cross = False,):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim // n_heads
        self.cross = cross

        self.dropout = dropout
        self.inner_dim = self.dim_head * n_heads

        self.norm = torch.nn.LayerNorm(dim)

        self.W_q = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_k = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_v = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_o = torch.nn.Linear(self.inner_dim, dim, bias = bias)

        self.mha = torch.nn.MultiheadAttention(dim,
                                               n_heads,
                                               dropout = dropout,
                                               batch_first=True)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, y = None, mask = None):
        """Input shape is (batch, seq_len, dim)"""
        x = self.norm(x)

        if self.cross and (not y is None):
            q, k, v = self.W_q(x), self.W_k(y), self.W_v(y)
        else:
            q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        
        output, _ = self.mha(q, k, v,
                             need_weights=False,
                             attn_mask=mask)

        output = self.W_o(output)

        return self.dropout(output)
    
class Transformer(torch.nn.Module):
    def __init__(self, 
                 dim = 512, 
                 depth = 4, 
                 heads = 8, 
                 dropout = 0.4,
                 positional_embedding = True,
                 context = None,
                 cross_context = None,
                 activation = torch.nn.GELU,
                 ema_decay = 0.996,
                 first_layer_norm = True,
                 cross = False):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.cross = cross
        self.context = context
        self.cross_context = cross_context

        self.ema_decay = ema_decay

        self.has_util_norm = False

        if first_layer_norm:
            self.norm = torch.nn.LayerNorm(dim)
            self.cross_norm = torch.nn.LayerNorm(dim)
        else:
            self.norm = torch.nn.Identity()
            self.cross_norm = torch.nn.Identity()

        if positional_embedding and (context is not None):
            self.pos_embedding = torch.nn.Parameter(torch.randn(1, context, dim))
        else:
            self.register_buffer("pos_embedding", torch.zeros(1, 1, dim))

        if positional_embedding and (cross_context is not None):
            self.pos_embedding_cross = torch.nn.Parameter(torch.randn(1, cross_context, dim))
        else:
            self.register_buffer("pos_embedding_cross", torch.zeros(1, 1, dim))
        

        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Attention(dim, n_heads = heads, dropout = dropout, cross = cross),
                MLP(dim, dim, dropout = dropout, activation = activation)
            ]))

    def forward(self,
                x,
                y = None,
                stop_at = None,
                pos_embedding = None,
                pos_embedding_cross = None,
                mask = None):
        if pos_embedding is None:
            pos_embedding = self.pos_embedding
        if (y is not None) and (pos_embedding_cross is None):
            y = self.cross_norm(y) + self.pos_embedding_cross
        x = self.norm(x) + pos_embedding

        for i, (attention, ff) in enumerate(self.layers):
            x = x + attention(x, y = y, mask = mask)
            x = x + ff(x)

            if (stop_at is not None) and (i >= (stop_at - 1)):
                break
        return x
    
class TransformerAutoencoder(torch.nn.Module):
    def __init__(self,
                 dim,
                 context,
                 context_compression = 0.5,
                 transformer_depth = 2,
                 transformer_heads = 2,
                 dropout = 0.1):
        super().__init__()
        self.dim = dim
        self.context = context

        if context_compression < 1:
            context_compression = int(context * context_compression)
        self.context_compression = context_compression

        self.encoder = Transformer(dim,
                                   depth = transformer_depth,
                                   heads = transformer_heads,
                                   context = context_compression,
                                   cross_context = context,
                                   dropout = dropout,
                                   cross = True)
        self.decoder = Transformer(dim,
                                   depth = transformer_depth,
                                   heads = transformer_heads,
                                   context = context,
                                   cross_context = context_compression,
                                   dropout = dropout,
                                   cross = True)
        self.encoder_mask_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.decoder_mask_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        
    def forward(self, x):
        encoded = self.initialize_encoded(x.shape[0])
        encoded = self.encoder(encoded, y = x)

        decoded = self.decoder_mask_token.repeat(x.shape[0],
                                                 self.context,
                                                 1)
        decoded = self.decoder(decoded, encoded)
        return encoded, decoded
    
    def initialize_encoded(self, batch_size, dummy = False):
        if dummy:
            return None
        return self.encoder_mask_token.repeat(batch_size, self.context_compression, 1)
    
    def initialize_decoded(self, batch_size, dummy = False):
        if dummy:
            return None
        return self.decoder_mask_token.repeat(batch_size, self.context, 1)

    
class GlobalWorkspaceIntegrator(torch.nn.Module):
    def __init__(self,
                 modules,
                 transformer_depth = 2,
                 transformer_heads = 2,
                 compression = 0.5,
                 dim = 256):
        super().__init__()
        self.all_modules = torch.nn.ModuleList(modules)
        self.n = len(modules)
        self.dim = dim
        embedders = []
        for i in range(len(modules)):
            out_dim = modules[i].out_dim
            embedders.append(torch.nn.Linear(out_dim, dim))

        self.embedders = torch.nn.ModuleList(embedders)
        self.compressor = TransformerAutoencoder(dim,
                                                 self.n,
                                                 context_compression = compression,
                                                 transformer_depth = transformer_depth,
                                                 transformer_heads = transformer_heads)
        self.latent_n = self.compressor.context_compression

    def forward(self, module_states, dummy = False, random = False):
        if dummy:
            if random:
                return (torch.randn(module_states[0].shape[0],
                                   self.latent_n,
                                   self.dim,
                                   device = module_states[0].device),
                        torch.randn(module_states[0].shape[0],
                                    self.n,
                                    self.dim,
                                    device = module_states[0].device))
            return None, None
        embed_states = []
        for i in range(len(module_states)):
            embed_states.append(self.embedders[i](module_states[i]))
        # B, N, D
        embed_states = torch.stack(embed_states, dim=1)
        encoded, decoded = self.compressor(embed_states)
        return encoded, decoded
    
    def modules_forward(self, x_split, c = None, embed = False):
        module_states = []
        for i in range(len(x_split)):
            module_states.append(self.all_modules[i](x_split[i],
                                                     c = c))
        if embed:
            module_states = [self.embedders[i](module_states[i]) for i in range(len(module_states))]
            module_states = torch.stack(module_states, dim = 1)
        return module_states
    
    
class ConditionedMLP(torch.nn.Module):
    def __init__(self,
                 encoder_sizes,
                 decoder_sizes,
                 conditioner_sizes,
                 activation = torch.nn.GELU,):
        super().__init__()

        encoder = [MLP(encoder_sizes[i], encoder_sizes[i + 1],
                       out_dim = encoder_sizes[i + 1],
                       activation = activation) for i in range(len(encoder_sizes) - 1)]
        decoder = [MLP(decoder_sizes[i], decoder_sizes[i + 1],
                       out_dim = decoder_sizes[i + 1],
                       activation = activation) for i in range(len(decoder_sizes) - 1)]
        conditioner = [MLP(conditioner_sizes[i], conditioner_sizes[i + 1],
                           out_dim = conditioner_sizes[i + 1],
                           activation = activation) for i in range(len(conditioner_sizes) - 1)]

        self.encoder = torch.nn.Sequential(*encoder)
        self.decoder = torch.nn.Sequential(*decoder)
        self.conditioner = torch.nn.Sequential(*conditioner)
        self.latent_dim = decoder_sizes[0]
        self.out_dim = decoder_sizes[-1]

    def forward(self, x, c = None):
        encoded = self.encoder(x)
        if c is not None:
            conditioned = self.conditioner(c)
            # reshape conditioned
            conditioned = conditioned.reshape(encoded.shape)
            encoded = torch.cat([encoded, conditioned],
                                dim = -1)
        else:
            # double the input
            encoded = torch.cat([encoded, encoded],
                                dim = -1)
        decoded = self.decoder(encoded)
        return decoded


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size = 3, 
                 stride = 1,
                 padding = 1,
                 activation = torch.nn.GELU()):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.conv_skip = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv_skip = torch.nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.activation(y)
        y = self.conv2(y)
        y = self.bn2(y)
        res = self.conv_skip(x)
        return self.activation(y + res)
    
class RescaleBlock(torch.nn.Module):
    """
    Simple rescaling + resnet block.
    """
    def __init__(self,
                 scale,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 mode = "nearest"):
        super().__init__()
        self.scale = scale
        self.resnet = ResnetBlock(in_channels, out_channels, kernel_size, stride, padding)

        self.mode = mode

    def forward(self, x):
        x = self.resnet(x)
        x_rescale = torch.nn.functional.interpolate(x,
                                                    scale_factor=self.scale,
                                                    mode=self.mode)
        return x_rescale

class ConditionedResnet(torch.nn.Module):
    def __init__(self,
                 out_dim,
                 scaling_factors,
                 encoder_channels,
                 decoder_channels,
                 conditioner_sizes,
                 activation = torch.nn.GELU,):
        super().__init__()

        conv_encoder = [RescaleBlock(1 / scaling_factors[i], encoder_channels[i], encoder_channels[i + 1])
                        for i in range(len(encoder_channels) - 1)]
        conv_decoder = [RescaleBlock(scaling_factors[i], decoder_channels[i], decoder_channels[i + 1])
                        for i in range(len(decoder_channels) - 1)]
        conditioner = [MLP(conditioner_sizes[i], conditioner_sizes[i + 1],
                           out_dim = conditioner_sizes[i + 1],
                           activation = activation) for i in range(len(conditioner_sizes) - 1)]

        self.encoder = torch.nn.Sequential(*conv_encoder)
        self.decoder = torch.nn.Sequential(*conv_decoder)
        self.conditioner = torch.nn.Sequential(*conditioner)

        self.latent_dim = 2 * encoder_channels[-1]
        self.out_dim = out_dim
        self.middle_linear = torch.nn.Linear(self.latent_dim,
                                             self.latent_dim // 2)

    def forward(self, x, c = None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        encoded = self.encoder(x)
        start_shape = encoded.shape
        encoded = encoded.reshape(encoded.shape[0], -1)
        if c is not None:
            conditioned = self.conditioner(c)
            conditioned = conditioned.reshape(encoded.shape)
            encoded = torch.cat([encoded, conditioned],
                                dim = -1)
        else:
            encoded = torch.cat([encoded, encoded],
                                dim = -1)
        encoded = self.middle_linear(encoded)
        decoded = self.decoder(encoded.reshape(start_shape))
        return decoded.squeeze(1)

# NOTES : in the only conditioned MLP scenario, GWSP learns faster when there are many modules, but otherwise is worse
# in the mixed MLP+resnet scenario, GWSP seems worse generally.

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from copy import deepcopy
    N_EPOCHS = 100
    STEPS_PER_EPOCH = 20
    BATCH_SIZE = 256

    GWSP_DIM = 16
    SYS_DIM = 256
    N_MODULES = 16
    CONTEXT_COMPRESS = 0.25
    GWSP_TOKENS = int(N_MODULES * CONTEXT_COMPRESS)
    torch.manual_seed(888)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    indices = rand_partition(N_MODULES, SYS_DIM, min_mult = GWSP_TOKENS)
    indices = [idx.item() for idx in indices]
    mlp_indices = indices[:(len(indices) // 2)]
    res_indices = indices[(len(indices) // 2):]

    losses = {"with_gwsp": [], "without_gwsp": []}

    # modules_mlp = [ConditionedMLP([n, 2 * n],
    #                           [4 * n, n],
    #                           [GWSP_DIM, 2 * n // GWSP_TOKENS]) for n in mlp_indices]
    # modules_conv = [ConditionedResnet(n,
    #                                   [n],
    #                                   [1, 16],
    #                                   [16, 1],
    #                                   [GWSP_DIM, 16 // GWSP_TOKENS]) for n in res_indices]
    # modules = modules_mlp + modules_conv
    modules = [ConditionedMLP([n, 2 * n],
                              [4 * n, n],
                              [GWSP_DIM, n, n, 2 * n // GWSP_TOKENS]) for n in indices]
    modules_clone = [deepcopy(m) for m in modules]

    gwsp = GlobalWorkspaceIntegrator(modules,
                                        dim = GWSP_DIM,
                                        compression = CONTEXT_COMPRESS)
    gwsp = gwsp.to(device)

    gwsp_clone = GlobalWorkspaceIntegrator(modules_clone,
                                           dim = GWSP_DIM,
                                           compression = CONTEXT_COMPRESS)
    gwsp_clone = gwsp_clone.to(device)

    # dummy forward to initialize the model
    with torch.no_grad():
        x = torch.rand(BATCH_SIZE, SYS_DIM,
                        device = device) * 2 * torch.pi
        x_split = x.split(indices, dim = 1)
        module_states = gwsp.modules_forward(x_split)
        encoded, decoded = gwsp(module_states)

        module_states_clone = gwsp_clone.modules_forward(x_split)
        encoded_clone, decoded_clone = gwsp_clone(module_states_clone)

    # oss = SynergisticKuramoto(SYS_DIM, K = 1.2,
    #                           block_indices = indices)
    oss = Lorenz96(SYS_DIM, F = 6.2)
    oss = oss.to(device)

    optimizer_s = torch.optim.Adam(gwsp.all_modules.parameters(),
                                lr = 1e-3)
    optimizer_g = torch.optim.Adam(list(gwsp.embedders.parameters())
                                + list(gwsp.compressor.parameters()),
                                lr = 1e-3)
    
    optimizer_s_clone = torch.optim.Adam(gwsp_clone.all_modules.parameters(),
                                         lr = 1e-3)

    module_losses = []
    module_losses_clone = []
    gwsp_losses = []

    pbar = tqdm(range(N_EPOCHS * STEPS_PER_EPOCH))
    for epoch in range(N_EPOCHS):
        x = torch.rand(BATCH_SIZE,
                        SYS_DIM, device = device) * 2 * torch.pi
        with torch.no_grad():
            encoded = gwsp.compressor.initialize_encoded(BATCH_SIZE)
            if isinstance(oss, SynergisticKuramoto):
                r, psi = oss.get_phase(x)
                r = r.mean().item()
                psi = psi.mean().item()

        for step_i in range(STEPS_PER_EPOCH):
            x = x % (2 * 3.14159)

            x_split = x.split(indices, dim = 1)
            module_states = gwsp.modules_forward(x_split, c = encoded)
            module_states_clone = gwsp_clone.modules_forward(x_split,
                                                             c = None)

            module_states_clean = [m.detach().clone().requires_grad_() for m in module_states]
            encoded, decoded = gwsp(module_states_clean)

            with torch.no_grad():
                x_next = oss(x.clone().detach())
                next_split = x_next.split(indices, dim = 1)
                next_module_states = gwsp.modules_forward(next_split,
                                                          c = encoded,
                                                          embed = True)

            module_loss = 0
            module_loss_clone = 0
            for i in range(len(module_states)):
                module_loss += torch.nn.functional.mse_loss(module_states[i],
                                                            next_split[i])
                module_loss_clone += torch.nn.functional.mse_loss(module_states_clone[i],
                                                                  next_split[i])
                
            optimizer_s.zero_grad()
            module_loss.backward()
            optimizer_s.step()

            optimizer_s_clone.zero_grad()
            module_loss_clone.backward()
            optimizer_s_clone.step()

            # skip first step since GWSP state isn't defined
            if step_i > 0:
                # predictive GWSP
                gwsp_loss = torch.nn.functional.mse_loss(decoded,
                                                         next_module_states)

                optimizer_g.zero_grad()
                gwsp_loss.backward()
                optimizer_g.step()
                gwsp_losses.append(gwsp_loss.item())

            encoded = encoded.detach()

            x = x_next.clone().detach().requires_grad_(True)
            module_losses.append(module_loss.item())
            module_losses_clone.append(module_loss_clone.item())
            
            pbar.update(1)
            pbar.set_description(f"E: {epoch}, M: {module_loss.item():.2f}")
    pbar.close()
    losses["with_gwsp"] = (module_losses, gwsp_losses)
    losses["without_gwsp"] = (module_losses_clone, None)

    plt.plot(losses["with_gwsp"][0], label = "With GWSP")
    plt.plot(losses["without_gwsp"][0], label = "Without GWSP")
    plt.legend()
    # plot gwsp loss on new axis
    if losses["with_gwsp"][1] is not None:
        plt.twinx()
        plt.plot(losses["with_gwsp"][1], color = "red", label = "GWSP Loss")
        plt.legend(loc = "upper left")
    plt.show()


