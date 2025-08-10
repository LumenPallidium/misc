import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from scipy.cluster.vq import kmeans, vq
from sympy.ntheory import factorint

def symlog(x):
    """Symmetric logarithm function."""
    return torch.sign(x) * torch.log1p(torch.abs(x))

def approximate_square_root(x):
    factor_dict = factorint(x)
    factors = []
    for key, item in factor_dict.items():
        factors += [key] * item
    factors = sorted(factors)

    a, b = 1, 1
    for factor in factors:
        if a <= b:
            a *= factor
        else:
            b *= factor
    return a, b

class SelfReferentialLayer(torch.nn.Module):
    def __init__(self, input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 self_rep_dim=16,
                 softmax=False):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if hidden_dim is None:
            hidden_dim = input_dim * 2

        self.embed = torch.nn.Linear(input_dim, hidden_dim)
        self.deembed = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.GELU()
        self.self_rep = torch.nn.Linear(hidden_dim, self_rep_dim)
        self.self_derep = torch.nn.Linear(self_rep_dim, hidden_dim)
        self.softmax = softmax

    def forward(self, x):
        h = self.embed(x)
        h = self.activation(h)
        x_hat = self.deembed(h)

        self_rep = self.self_rep(h.detach())
        if self.softmax:
            self_rep = torch.softmax(self_rep, dim=-1)
        else:
            self_rep = self.activation(self_rep)
        self_derep = self.self_derep(self_rep)

        # drive to represent h well
        self_rep_loss = torch.mean((self_derep - h.detach()) ** 2)
        # drive h to be close to self_derep
        commit_loss = torch.mean((h - self_derep.detach()) ** 2)

        return x_hat, self_rep_loss, commit_loss

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim = None, output_dim = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if hidden_dim is None:
            hidden_dim = input_dim * 2

        self.norm = torch.nn.LayerNorm(input_dim)
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
class SimpleVAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=None, activation=torch.nn.GELU):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2

        self.encoder = torch.nn.Linear(input_dim, 2 * hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim)
        self.activation = activation()

    def encode(self, x):
        h = self.encoder(x)
        h = self.activation(h)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar
    
    def embed(self, x):
        """Embed input x into latent space."""
        mu, _ = self.encode(x)
        return mu
        
    def forward(self, x, stochastic=True):
        mu, logvar = self.encode(x)
        if not stochastic:
            return self.decode(mu), mu, logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_hat = self.decoder(z)
        return x_hat, z, mu, logvar
    
    def loss_function(self, x, x_hat, mu, logvar):
        recons_loss = torch.mean((x - x_hat) ** 2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recons_loss, kl_loss
    
class DeepMLP(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim=None,
                 residual=True,
                 num_layers=3):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.residual = residual
        self.embed = torch.nn.Linear(input_dim, hidden_dim)
        layers = [MLP(hidden_dim) for _ in range(num_layers - 2)]
        self.layers = torch.nn.ModuleList(layers)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            activity = layer(x)
            if self.residual:
                x = x + activity
            else:
                x = activity
        x = self.output_layer(x)
        return x

class FactorDiscriminator(torch.nn.Module):
    def __init__(self, input_dim,
                 hidden_dim=None,
                 mlp_depth=3,
                 output_dim=1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2

        self.network = DeepMLP(input_dim,
                               hidden_dim,
                               output_dim=output_dim,
                               num_layers=mlp_depth)

    def forward(self, x):
        h = self.network(x)
        return torch.sigmoid(h)
    
    def get_disc_loss(self, x):
        batch_size, dim = x.size()
        # efficient way to generate a batch of permutations
        permutations = torch.argsort(torch.rand(batch_size, dim), dim=-1)
        x_permuted = x.clone()[torch.arange(batch_size)[:, None], permutations]

        loss = torch.log(self.forward(x) + 1e-8) + \
                torch.log(1 - self.forward(x_permuted) + 1e-8)

        # technically need to average!
        return 0.5 * loss.mean()
    
    def get_gen_loss(self, x):
        """From FactorVAE, based on density ratio trick.
        KL(q||q') = E_q[log(q/q')] ~ E_q[log(D(x)/(1-D(x)))]"""
        y = self.forward(x)
        loss = torch.log(y + 1e-8) - torch.log(1 - y + 1e-8)
        loss = -loss.mean()
        return loss
  
    
class EntangledFactorGenerator(torch.nn.Module):
    def __init__(self,
                 out_dim,
                 n_factors = 16,
                 factor_dims = 4,
                 max_var = 5,
                 sparsity = 0.6):
        super().__init__()
        self.out_dim = out_dim
        self.n_factors = n_factors
        self.factor_dims = factor_dims
        self.sparsity = sparsity
        self.register_buffer("factor_means", torch.randn(n_factors, factor_dims))
        self.register_buffer("factor_stds", torch.rand(n_factors, factor_dims) * max_var)
        self._generate_mix_weights()

    def forward(self, factors, factor_effects):
        factors = factors + factor_effects
        return self.represent_factors(factors)
    
    def _generate_mix_weights(self):
        """Generate random weights for mixing outer products."""
        mix_weights = torch.randn(self.out_dim, self.out_dim) * 0.1
        # Apply sparsity
        mask = torch.rand(self.out_dim, self.out_dim) > self.sparsity
        mix_weights[mask] = 0
        self.register_buffer("mix_weights", mix_weights)

        proj_weights = torch.randn(self.n_factors, self.factor_dims, self.out_dim)
        # Apply sparsity
        mask = torch.rand(self.n_factors, self.factor_dims, self.out_dim) > self.sparsity
        proj_weights[mask] = 0
        self.register_buffer("proj_weights", proj_weights)

    def full_sample(self, batch_size = 1):
        """Sample pure factors and generate their entangled representation."""
        factors = self.sample_factors(batch_size)
        return self.represent_factors(factors)

    def represent_factors(self, factors):
        """Generate a high dimensional entangled representation of factors."""
        result = torch.einsum("...nd,ndo->...no",
                              factors,
                              self.proj_weights)
        result = symlog(result.sum(dim = -2))
        result = torch.einsum("...o,od->...d",
                              result,
                              self.mix_weights)
        return result
    
    def sample_factors(self, batch_size = 1):
        """Sample pure factors"""
        factors = torch.randn(batch_size, self.n_factors, self.factor_dims,
                              device=self.factor_means.device)
        factors = factors * self.factor_stds + self.factor_means
        return factors   

class TensorEntangledFactorGenerator(torch.nn.Module):
    def __init__(self,
                 n_factors = 8,
                 factor_dims = 4,
                 max_outer_products = 4,
                 max_var = 5):
        super().__init__()
        assert n_factors % max_outer_products == 0, "n_factors must be divisible by max_outer_products"
        self.n_factors = n_factors
        self.factor_dims = factor_dims
        self.max_outer_products = max_outer_products
        self.out_dim = factor_dims ** max_outer_products
        self.n_linear_layers = n_factors // max_outer_products
        self.register_buffer("factor_means", torch.randn(n_factors, factor_dims))
        self.register_buffer("factor_stds", torch.rand(n_factors, factor_dims) * max_var)
        self.register_buffer("mix_weights", torch.randn(self.n_linear_layers, self.out_dim, self.out_dim))

    def forward(self, factors, factor_effects):
        factors = factors + factor_effects
        return self.represent_factors(factors)
    
    def full_sample(self, batch_size = 1):
        """Sample pure factors and generate their entangled representation."""
        factors = self.sample_factors(batch_size)
        return self.represent_factors(factors)

    def represent_factors(self, factors):
        """Generate a high dimensional entangled representation of factors."""
        batch_size = factors.size(0)
        factor_chunks = factors.chunk(self.n_linear_layers, dim=1)
        result = 0
        for i, chunk in enumerate(factor_chunks):
            outer_prod = chunk[:, 0, :]
            for j in range(1, chunk.size(1)):
                outer_prod = torch.einsum("bi,bj->bij", outer_prod, chunk[:, j, :])
                # flatten
                outer_prod = outer_prod.view(batch_size, -1)
            # normalize outer product
            outer_prod = outer_prod / self.out_dim
            result += torch.einsum("bi,ij->bj",
                                   outer_prod,
                                   self.mix_weights[i])
        return result
    
    def sample_factors(self, batch_size = 1):
        """Sample pure factors"""
        factors = torch.randn(batch_size, self.n_factors, self.factor_dims,
                              device=self.factor_means.device)
        factors = factors * self.factor_stds + self.factor_means
        return factors

def smooth_losses(losses, smooth_n = 10):
    losses = np.array(losses)
    conv_kernel = np.ones(smooth_n) / smooth_n
    smoothed = np.convolve(losses, conv_kernel, mode="same")
    return smoothed

def total_correlation(data,
                      sizes = [1024],
                      make_fig = False,
                      make_histogram = False):
    # convert data to numpy array if it's a tensor
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if isinstance(sizes, int):
        sizes = [sizes]

    corrs = []
    for size in sizes:
        centroids, _ = kmeans(data, size)
        code, _ = vq(data, centroids)
        # normalize code to a distribution
        labels, counts = np.unique(code, return_counts=True)
        distribution = counts / counts.sum()

        batch_size, dim = data.shape
        permutations = np.argsort(np.random.rand(batch_size, dim), axis=-1)
        data_permuted = data[np.arange(batch_size)[:, None], permutations]
        alt_code, _ = vq(data_permuted, centroids)
        alt_labels, alt_counts = np.unique(alt_code, return_counts=True)
        # check for missing labels and add 0
        alt_distribution = np.zeros(size)
        alt_distribution[alt_labels] = alt_counts
        where_zero = alt_distribution == 0
        if where_zero.any():
            print("\tWarning: some labels missing in the alt distribution")
            distribution_cut = distribution[~where_zero]
            distribution_cut = distribution_cut / distribution_cut.sum()
            alt_distribution_cut = alt_distribution[~where_zero]
        else:
            distribution_cut = distribution
            alt_distribution_cut = alt_distribution
        alt_distribution_cut = alt_distribution_cut / alt_distribution_cut.sum()
        # calculate total correlation
        total_corr = np.sum(distribution_cut * (np.log(distribution_cut) - np.log(alt_distribution_cut)))
        corrs.append(total_corr)

    corrs = np.array(corrs)

    if make_fig:
        # do log regression to estimate the scaling of total correlation with size
        log_sizes = np.log(sizes)
        log_corrs = np.log(corrs)
        A = np.vstack([log_sizes, np.ones_like(log_sizes)]).T
        m, c = np.linalg.lstsq(A, log_corrs, rcond=None)[0]
        fig, ax = plt.subplots()
        # log plotting
        ax.plot(log_sizes, log_corrs, 'o', label='Data')
        ax.plot(log_sizes, c + m * log_sizes , label='Log Regression')
        ax.set_xlabel('log(Size)')
        ax.set_ylabel('log(Total Correlation)')
        ax.legend()
    if make_histogram:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        l1, l2 = approximate_square_root(sizes[0])
        distribution_square = distribution.reshape(l1, l2)
        alt_distribution_square = alt_distribution.reshape(l1, l2)
        ax[0].imshow(distribution_square, cmap="viridis")
        ax[0].set_title("Distribution")
        ax[1].imshow(alt_distribution_square, cmap="viridis")
        ax[1].set_title("Alternative Distribution")

    return corrs[-1]

def evaluate_model_tc(generator, model, samples = 10000, batch_size = 128,
                      size = 1024, make_fig = False, make_histogram = False):
    """Evaluate total correlation of a model's representations."""
    model = model.eval()

    n_batches = samples // batch_size
    representations = []
    
    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Evaluating TC"):
            x = generator.full_sample(batch_size=batch_size)
            x = x.to(next(model.parameters()).device)
            rep = model.embed(x)
            representations.append(rep.cpu().numpy())
    
    representations = np.concatenate(representations, axis=0)
    return total_correlation(representations, sizes=[size],
                             make_fig=make_fig,
                             make_histogram=make_histogram)

def plot_factor_manifold(generator, models,
                         pts_per_factor = 100):
    from sklearn.decomposition import PCA
    if isinstance(models, torch.nn.Module):
        models = [models]
    n_models = len(models)
    fig, ax = plt.subplots(1, n_models,
                           figsize=(4 * n_models,
                                    6))
    n_pts = pts_per_factor ** 3
    line = torch.linspace(-1, 1, pts_per_factor,
                          device = generator.factor_means.device)[:,None]

    with torch.no_grad():
        factors = (generator.sample_factors(1)[None, ...]).repeat(3, pts_per_factor, 1, 1)
        # make 3 lines in factor space
        for i in range(3):
            factors[i, :, i * generator.factor_dims, :] += line
        representations = generator.represent_factors(factors)
        embeddings = [model.embed(representations) for model in models]
        embeddings = [emb.cpu().numpy() for emb in embeddings]
        #variances = [np.var(emb, axis=2).mean() / np.var(emb, axis=(0, 1)).mean() for emb in embeddings]
    
    # use first embedding dimension as color
    cmaps = ["Blues", "Reds", "Greens",]
    for i, emb in enumerate(embeddings):
        # flatten the embeddings
        emb = emb.reshape(emb.shape[0] * emb.shape[1], -1)
        # do pca
        pca = PCA(n_components=2)
        embedding_i = pca.fit_transform(emb)
        
        for j in range(3):
            ax[i].scatter(embedding_i[j*pts_per_factor:(j+1)*pts_per_factor, 0],
                          embedding_i[j*pts_per_factor:(j+1)*pts_per_factor, 1],
                          c=np.linspace(0, 1, pts_per_factor),
                          cmap=cmaps[j], s=5)
        ax[i].set_title(f"Model {i} Factor Manifold")
        ax[i].set_xlabel("PCA 1")
        ax[i].set_ylabel("PCA 2")

    plt.tight_layout()
    return fig, ax

def train_factorvae(generator,
                    vae_hidden_dim=128,
                    disc_hidden_dim=128,
                    disc_depth=3,
                    disc_weight = 0.1,
                    kl_weight = 1,
                    n_steps=5000,
                    batch_size = 128,
                    vae_lr = 1e-3,
                    disc_lr = 1e-5,
                    device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = generator.out_dim
    vae = SimpleVAE(input_dim=dim,
                    hidden_dim=vae_hidden_dim).to(device)
    discrimator = FactorDiscriminator(input_dim=dim,
                                      hidden_dim=disc_hidden_dim,
                                      mlp_depth=disc_depth).to(device)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=vae_lr)
    optimizer_disc = torch.optim.Adam(discrimator.parameters(), lr=disc_lr)

    pbar = tqdm(range(n_steps))
    losses = []
    disc_losses = []
    for step in range(n_steps):
        with torch.no_grad():
            x = generator.full_sample(batch_size=batch_size)
            x = x.to(device)

        # train VAE
        x_hat, z, mu, logvar = vae(x)
        recons_loss, kl_loss = vae.loss_function(x, x_hat, mu, logvar)

        factor_loss = discrimator.get_gen_loss(z)
        loss = recons_loss + kl_weight * kl_loss + disc_weight * factor_loss
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer_vae.step()

        # train discriminator - paper says to use a new batch
        with torch.no_grad():
            x_new = generator.full_sample(batch_size=batch_size)
            x_new = x_new.to(device)
            _, z_alt, _, _ = vae(x_new)
        disc_loss = discrimator.get_disc_loss(z_alt)
        optimizer_disc.zero_grad()
        disc_loss.backward()
        optimizer_disc.step()

        losses.append(loss.item())
        disc_losses.append(disc_loss.item())

        pbar.set_description(f"Loss: {loss.item():.1f}, "
                             f"Disc Loss: {disc_loss.item():.1f}")
        pbar.update(1)
    pbar.close()

    fig, ax = plt.subplots()
    ax.plot(smooth_losses(losses), label="VAE Loss")
    ax2 = ax.twinx()
    ax2.plot(smooth_losses(disc_losses),
             c ="orange",
             label="Discriminator Loss")
    ax.set_xlabel("Step")
    ax.legend()
    ax2.set_ylabel("Discriminator Loss")
    ax2.legend(loc="upper left")
    fig.tight_layout()

    return vae, discrimator

def train_selfcompressor(generator,
                         hidden_dim = 128,
                         n_steps = 5000,
                         batch_size = 128,
                         commit_preweight = 1,
                         softmax = True,
                         make_graphs = True,
                         device = None):
    input_dim = generator.out_dim
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelfReferentialLayer(input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 softmax=softmax).to(device)
    model2 = deepcopy(model)
    # untrained model for comparison
    model0 = deepcopy(model)

    generator = generator.to(device)
    model = model.to(device)
    model2 = model2.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3)
    optimizer2 = torch.optim.Adam(model2.parameters(),
                                  lr=1e-3)
    
    losses1 = []
    losses2 = []
    self_rep_losses1 = []
    self_rep_losses2 = []

    pbar = tqdm(range(n_steps))
    for step in range(n_steps):
        x = generator.full_sample(batch_size=batch_size)
        x = x.to(device)
        x_hat, self_rep_loss, commit_loss = model(x)
        recons_loss = torch.mean((x - x_hat) ** 2)
        commit_weight = commit_preweight * torch.clamp(1 / (self_rep_loss.detach() + 1e-6), max=1)
        loss = recons_loss + self_rep_loss + commit_weight * commit_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        x_hat2, self_rep_loss2, commit_loss2 = model2(x)
        recons_loss2 = torch.mean((x - x_hat2) ** 2)
        loss2 = recons_loss2 + self_rep_loss2
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        losses1.append(recons_loss.item())
        losses2.append(recons_loss2.item())
        self_rep_losses1.append(self_rep_loss.item())
        self_rep_losses2.append(self_rep_loss2.item())

        pbar.set_description(f"Loss: {recons_loss.item():.1f}/{recons_loss2.item():.1f}, "
                             f"Self Rep Loss: {self_rep_loss.item():.1f}/{self_rep_loss2.item():.1f}, ")
        pbar.update(1)
    pbar.close()

    if make_graphs:
        fig, ax = plt.subplots()
        ax.plot(smooth_losses(losses1),
                label="Loss 1")
        ax.plot(smooth_losses(losses2),
                label="Loss 2")
        ax2 = ax.twinx()
        ax2.plot(smooth_losses(self_rep_losses1),
                label="Self Rep Loss 1", linestyle="--")
        ax2.plot(smooth_losses(self_rep_losses2),
                label="Self Rep Loss 2", linestyle="--")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.show()

        # compare representations
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(model.embed.weight.detach().cpu().numpy(),
                    aspect="auto", cmap="viridis")
        ax[0].set_title("Model 1 Representations")
        ax[1].imshow(model2.embed.weight.detach().cpu().numpy(),
                    aspect="auto", cmap="viridis")
        ax[1].set_title("Model 2 Representations")
        plt.show()

    return model0, model, model2

if __name__ == "__main__":
    input_dim = 256  # Example input dimension
    corr_size = 1024
    generator = EntangledFactorGenerator(input_dim)
    model_null, model_sr, model_alt = train_selfcompressor(generator, commit_preweight=1)
    model_vae, disc = train_factorvae(generator)
    for name, model in zip(["FactorVAE", "Untrained", "Self-Referential", "Alternative"],
                           [model_vae, model_null, model_sr, model_alt]):
        tc = evaluate_model_tc(generator, model, size=corr_size)
        print(f"{name} Model Total Correlation: {tc:.2f}")

    #fig, ax = plot_factor_manifold(generator, [model0, model, model2])