import torch
from torch.nn.functional import mse_loss

class MLP(torch.nn.Module):
    """
    Classic multi-layer perceptron (MLP) model.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_mults = 3,
                 activation = torch.nn.GELU()):
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
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if isinstance(hidden_mults, int):
            hidden_mults = [hidden_mults]

        layers = []
        in_dim = input_dim
        for mult in hidden_mults:
            layers.append(torch.nn.Linear(in_dim, int(in_dim * mult)))
            layers.append(activation)
            in_dim = int(in_dim * mult)
        layers.append(torch.nn.Linear(in_dim, output_dim))
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
                 alpha = 0.01):
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

        if isinstance(hidden_mults, int):
            hidden_mults = [hidden_mults]
        self.hidden_mults = hidden_mults

        self.embedder = torch.nn.Linear(input_dim * delay, hidden_dim)
        self.encoder = MLP(hidden_dim, bottleneck_dim, hidden_mults)

        if self.use_decoder:
            hidden_mults = hidden_mults[::-1]
            self.decoder = MLP(bottleneck_dim, hidden_dim, hidden_mults)
            # note deembedder does not reconstruct whole sequence, only the last element
            self.deembedder = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.embedder(x)
        x = self.encoder(x)
        if self.use_decoder:
            x_hat = self.decoder(x)
            x_hat = self.deembedder(x_hat)
        else:
            x_hat = None
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

        #TODO : y_t is ill-conditioned sometimes
        A = (y_t1 @ torch.pinverse(y_t)) @ y_t
        # frobenius norm of y_t1 - A
        loss = torch.norm(y_t1 - A) ** 2

        if self.use_decoder:
            rec_loss = 0
            # only get last element of x_t_hat
            rec_loss += mse_loss(x_t[:, -1, :], x_t_hat)
            rec_loss += mse_loss(x_t1[:, -1, :], x_t1_hat)
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
            x[:, 0].add_(dt * ((x[:, 0] ** 3) / 3 + x[:, 0] - x[:, 1] + self.I))
            x[:, 1].add_(dt * self.c * (x[:, 0] - self.b * x[:, 1] + self.a))

            x_t.append(x.clone())
        x_t = torch.stack(x_t, dim = 1)

        return x_t # batch, T, 2
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    DELAY = 16
    N_STEPS = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LKIS(2, DELAY, 8, use_decoder = True)
    model.to(device)
    ds = FitzHughNagumoDS()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    losses = []
    for i in tqdm(range(N_STEPS)):
        optimizer.zero_grad()

        sample = ds.sample(T = DELAY + 1).to(device)
        x_t = sample[:, :-1, :].detach().clone()
        x_t1 = sample[:, 1:, :].detach().clone().requires_grad_(True)

        loss = model.get_loss(x_t, x_t1)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    plt.plot(losses)



