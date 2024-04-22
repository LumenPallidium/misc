import torch
from torch.nn.functional import mse_loss


class ComplexGELU(torch.nn.Module):
    """
    Complex GELU activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        new_real = torch.nn.functional.gelu(x.real)
        # need to do this to avoid in-place operation
        y = torch.complex(new_real, x.imag)
        return y

class MLP(torch.nn.Module):
    """
    Classic multi-layer perceptron (MLP) model.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_mults = 3,
                 activation = ComplexGELU(),
                 complex = True):
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the input data.
        output_dim : int
            The dimension of the output data.
        hidden_mults : int or list of int, optional
            The multiplier for the hidden layers. If int, the same multiplier is used for all hidden layers.
            If list, the multipliers are used in the order given. Defaults to 3.
        activation : torch.nn.Module, optional
            The activation function to use. Defaults to GELU.
        complex : bool, optional
            Whether to use complex numbers. Defaults to True.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if isinstance(hidden_mults, int):
            hidden_mults = [hidden_mults]

        dtype = torch.complex64 if complex else None

        layers = []
        in_dim = input_dim
        for mult in hidden_mults:
            layers.append(torch.nn.Linear(in_dim, int(in_dim * mult),
                                          dtype = dtype))
            layers.append(activation)
            in_dim = int(in_dim * mult)
        layers.append(torch.nn.Linear(in_dim, output_dim,
                                      dtype = dtype),)
        self.layers = torch.nn.Sequential(*layers)
        

    def forward(self, x):
        return self.layers(x)
    
class LKIS(torch.nn.Module):
    """
    The learning Koopman invariant subspaces (LKIS) model.
    Uses deep learning to learn the the measurement operators in Koopman theory.
    From the paper:
    https://arxiv.org/pdf/1710.04340.pdf
    """
    def __init__(self,
                 input_dim,
                 delay,
                 hidden_dim,
                 bottleneck_dim = None,
                 hidden_mults = 3,
                 use_decoder = True,
                 alpha = 0.01,
                 complex =True):
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the input data.
        delay : int
            The delay in the input data i.e. number of time samples
        hidden_dim : int
            The dimension of the hidden layer after timeseries embedding.
        bottleneck_dim : int, optional
            The dimension of the bottleneck layer. If not given, defaults to hidden_dim.
        hidden_mults : int or list of int, optional
            The multiplier for the hidden layers. If int, the same multiplier is used for all hidden layers.
            If list, the multipliers are used in the order given.
        use_decoder : bool, optional
            Whether to use a decoder to reconstruct the input data. Defaults to True.
        alpha : float, optional
            The weight of the reconstruction loss. Defaults to 0.01.
        complex : bool, optional
            Whether to use complex numbers. Defaults to True.
        """
        super().__init__()
        self.input_dim = input_dim
        self.delay = delay
        self.hidden_dim = hidden_dim
        self.use_decoder = use_decoder
        if bottleneck_dim is None:
            bottleneck_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.alpha = alpha
        self.complex = complex
        if complex:
            activation = ComplexGELU()
            embed_mult = 2
        else:
            activation = torch.nn.GELU()
            embed_mult = 1

        assert bottleneck_dim % delay == 0, "hidden_dim must be divisible by delay"

        if isinstance(hidden_mults, int):
            hidden_mults = [hidden_mults]
        self.hidden_mults = hidden_mults

        self.embedder = torch.nn.Linear(input_dim * delay, hidden_dim * embed_mult)
        self.encoder = MLP(hidden_dim, bottleneck_dim,
                           hidden_mults, activation = activation,
                           complex = complex)

        if self.use_decoder:
            hidden_mults = hidden_mults[::-1]
            self.decoder = MLP(bottleneck_dim, hidden_dim,
                               hidden_mults, activation = activation,
                               complex = complex)
            self.deembedder = torch.nn.Linear(hidden_dim * embed_mult, input_dim * delay)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.embedder(x)

        if self.complex:
            # reshape and view as complex numbers
            x = x.view(batch_size, -1, 2)
            x = torch.view_as_complex(x)

        x = self.encoder(x)
        if self.use_decoder:
            x_hat = self.decoder(x)

            if self.complex:
                x_hat = torch.view_as_real(x_hat)
                x_hat = x_hat.view(batch_size, -1)

            x_hat = self.deembedder(x_hat)
            x_hat = x_hat.view(batch_size, self.delay, -1)
        else:
            x_hat = None

        # unstack x
        x = x.view(batch_size, self.delay, -1)

        return x, x_hat
    
    def get_loss(self, x_t, x_t1):
        """
        Given a pair of sequences x_t and x_t1, computes the loss of the model.

        #TODO write more here

        Parameters
        ----------
        x_t : torch.Tensor
            The first sequence of the pair.
        x_t1 : torch.Tensor
            The second sequence of the pair. Should be one step delayed from x_t.
        """
        with torch.no_grad():
            y_t, x_t_hat = self.forward(x_t)
        y_t1, x_t1_hat = self.forward(x_t1)

        y_t_inv = torch.pinverse(y_t)

        A = torch.einsum("bdi,bjd->bij", y_t1, y_t_inv).mean(dim = 0)
        Ay = torch.einsum("ij, bdj -> bdi", A, y_t)
        # frobenius norm of y_t1 - Ay
        loss = torch.linalg.matrix_norm(y_t1 - Ay).sum()

        if self.use_decoder:
            rec_loss = 0
            # only get last element of x_t_hat
            rec_loss += mse_loss(x_t, x_t_hat)
            rec_loss += mse_loss(x_t1, x_t1_hat)
            loss += self.alpha * rec_loss
        return loss

    
class FitzHughNagumoDS:
    """
    The FitzHugh-Nagumo equation is a simplified model of the electrical activity of a neuron.

    Here is used as a simple dynamical system to test the Koopman learning model.
    """
    def __init__(self, a = 0.7, b = 0.8, c = 0.08, I = 0.8):
        self.a = a
        self.b = b
        self.c = c
        self.I = I

    def sample(self, batch_size = 256, T = 100, dt = 0.01, x0 = None):
        if x0 is None:
            x0 = torch.randn(batch_size, 2)
        x = x0
        x_t = [x.clone()]
        for t in range(T - 1):
            x[:, 0].add_(dt * (-(x[:, 0] ** 3) / 3 + x[:, 0] - x[:, 1] + self.I))
            x[:, 1].add_(dt * self.c * (x[:, 0] - self.b * x[:, 1] + self.a))

            x_t.append(x.clone())
        x_t = torch.stack(x_t, dim = 1)

        return x_t # batch, T, 2

#TODO : cleanup below
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    DELAY = 16
    N_STEPS = 500
    BATCH_SIZE = 512
    HIDDEN_DIM = 32
    HIDDEN_MULTS = [2, 3, 2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LKIS(2,
                 DELAY,
                 HIDDEN_DIM * DELAY,
                 hidden_mults = HIDDEN_MULTS,
                 use_decoder = True)
    model.to(device)
    ds = FitzHughNagumoDS()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    losses = []
    for i in tqdm(range(N_STEPS)):
        optimizer.zero_grad()

        sample = ds.sample(T = DELAY + 1,
                           batch_size = BATCH_SIZE).to(device)
        x_t = sample[:, :-1, :].detach().clone()
        x_t1 = sample[:, 1:, :].detach().clone().requires_grad_(True)

        loss = model.get_loss(x_t, x_t1)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    with torch.no_grad():
        sample = ds.sample(T = (DELAY + 1) * BATCH_SIZE,
                           batch_size = 1).to(device)
        sample = sample.view(BATCH_SIZE, DELAY + 1, 2)
        x_t = sample[:, :-1, :].detach().clone()
        x_t1 = sample[:, 1:, :].detach().clone()

        y_t, x_t_hat = model.forward(x_t)
        y_t1, x_t1_hat  = model.forward(x_t1)

        # this is the Koopman operator
        y_t_inv = torch.pinverse(y_t)
        A = torch.einsum("bdi,bjd->bij", y_t1, y_t_inv)
        mean_var = A.var(dim = 0).mean()
        A = A.mean(dim = 0)

        y_t = y_t.reshape(DELAY * BATCH_SIZE, HIDDEN_DIM)
        y_past = y_t[:DELAY * BATCH_SIZE // 2, :].clone()
        y_future = y_t[DELAY * BATCH_SIZE // 2:, :].clone()

        A_t = torch.linalg.matrix_power(A, DELAY // 2)
        y_future_hat = torch.einsum("ij, bj -> bi", A_t, y_past)

        x_future_hat = model.decoder(y_future_hat.reshape(BATCH_SIZE // 2, DELAY * HIDDEN_DIM))
        if model.complex:
            x_future_hat = torch.view_as_real(x_future_hat)
            x_future_hat = x_future_hat.view(BATCH_SIZE, -1)
        x_future_hat = model.deembedder(x_future_hat)

        x_t_plt = x_t.view(DELAY * BATCH_SIZE, 2)[DELAY * BATCH_SIZE // 2:, :]
        x_t_plt = x_t_plt.detach().cpu().numpy()
        x_future_hat = x_future_hat.view(BATCH_SIZE // 2 * DELAY, 2)
        x_future_hat = x_future_hat.detach().cpu().numpy()
        
    log_losses = np.log(losses)
    fig, ax = plt.subplots(1, 4, figsize = (10, 5))
    ax[0].plot(log_losses)
    ax[0].set_title("Log Loss")
    ax[1].imshow(A.real.detach().cpu().numpy())
    ax[1].set_title("Koopman Matrix Approximation")
    ax[2].imshow(A.imag.detach().cpu().numpy())


    #TODO : this is totally off
    ax[3].plot(x_t_plt[:, 0],
               x_t_plt[:, 1],
               label = "$x_t$",
               alpha = 0.8)
    ax[3].plot(x_future_hat[:, 0],
               x_future_hat[:, 1],
               label = "$\hat{x_t}$",
               alpha = 0.5)
    ax[3].set_title("$x_t$ vs $\hat{x_t}$")

    ax[3].legend()
    plt.tight_layout()



