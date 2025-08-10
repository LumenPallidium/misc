import os
import torch
from torch.nn.functional import conv2d, interpolate, mse_loss, pad, max_pool2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ID_KERNEL = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
AVG_KERENL = (1/9) * torch.ones(3, 3)
SOBEL_X = (1/8) * torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
SOBEL_Y = SOBEL_X.clone().transpose(0, 1)
LAPLACIAN = (1 /4) * torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]])

FULL_FILTER = torch.stack([ID_KERNEL,
                           #AVG_KERENL,
                           SOBEL_X,
                           SOBEL_Y,
                           #LAPLACIAN,
                           ],
                           dim = 0)[:, None, :, :]


def plot_stack(im_list, time_run = 6, ax_labels = None):
    len_ =len(im_list)
    fps = len_ / time_run
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=1800)
    len_ =len(im_list)

    def update_plot(num, data_list, plots, ax):
        for plot in plots:
            plot.set_data(data_list[num])
        plt.suptitle(f"t = {(num/len_)*time_run:.2f} s")

        return plots

    fig, ax  = plt.subplots(1, 1)
    fig.set_tight_layout(True)

    plots = [ax.imshow(im_list[0])]
    plots = tuple(plots)
    line_ani = animation.FuncAnimation(fig, update_plot, len_ - 1,
                                       fargs = (im_list, plots, ax), interval = 50, blit = True)
    os.makedirs("figures", exist_ok=True)
    line_ani.save("figures/nca.mp4", writer=writer)

class NCAMLPCol(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim = None,
                 hidden_size = 256,
                 activation = torch.nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        if not isinstance(hidden_size, list):
            hidden_size = [hidden_size]
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        layers = []
        prev_dim = in_dim
        for i in range(len(hidden_size)):
            layers.append(torch.nn.Linear(prev_dim,
                                          hidden_size[i]))
            prev_dim = hidden_size[i]
        self.layers = torch.nn.ModuleList(layers)
        if out_dim is not None:
            self.final_layer = torch.nn.Linear(prev_dim, out_dim)
            self.final_layer.weight.data.fill_(0)
        else:
            self.final_layer = torch.nn.Identity()
        self.activation = activation()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return self.final_layer(x)
    
class NCAConvCol(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim = None,
                 hidden_size = 256,
                 activation = torch.nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        if not isinstance(hidden_size, list):
            hidden_size = [hidden_size]
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        layers = []
        prev_dim = in_dim
        for i in range(len(hidden_size)):
            layers.append(torch.nn.Conv2d(prev_dim,
                                          hidden_size[i],
                                          kernel_size=1,
                                          padding=0))
            prev_dim = hidden_size[i]
        self.layers = torch.nn.ModuleList(layers)
        if out_dim is not None:
            self.final_layer = torch.nn.Conv2d(prev_dim, out_dim, kernel_size=1)
            # set to 0 to begin
            self.final_layer.weight.data.fill_(0)
        else:
            self.final_layer = torch.nn.Identity()
        self.activation = activation()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return self.final_layer(x)
    
class FixedKernelNCA(torch.nn.Module):
    def __init__(self,
                 dim,
                 kernel,
                 hidden_dim = 64,
                 visible_channels = 4,
                 alpha_channel_idx = 3,
                 step_size = 1,
                 residual = True,
                 convolutional = False,
                 mask_prob = 0.5,
                 h = 30, w = 30):
        super().__init__()
        self.dim = dim
        self.hidden = hidden_dim
        self.visible_channels = visible_channels
        self.alpha_channel_idx = alpha_channel_idx
        self.residual = residual
        self.convolutional = convolutional
        self.h = h
        self.w = w
        self.step_size = step_size
        self.mask_prob = mask_prob

        self.kernel = kernel
        self.col_dim = kernel.shape[0]
        if convolutional:
            self.col_net = NCAConvCol(self.col_dim,
                                      hidden_size = hidden_dim,
                                      out_dim = self.dim)
        else:
            # use MLP
            self.col_net = NCAMLPCol(self.col_dim,
                                hidden_size = hidden_dim,
                                out_dim = self.dim)
            
        
    def forward(self, grid, stochastic = True):
        batch_size = grid.shape[0]
        channel_dim = 1
        pad_grid = pad(grid, (1, 1, 1, 1), mode = "constant")
        kernel_grid = conv2d(pad_grid,
                             FULL_FILTER,
                             padding = "valid",
                             groups = 1)
        if not self.convolutional:
            kernel_grid = kernel_grid.transpose(1, -1)
            channel_dim = -1
        out_grid = self.col_net(kernel_grid)

        if stochastic:
            mask = torch.rand(batch_size, self.h, self.w,)
            mask = (mask < self.mask_prob).unsqueeze(channel_dim)
            out_grid = out_grid * mask

        if not self.convolutional:
            out_grid = out_grid.transpose(-1, 1)
        if self.residual:
            out_grid *= self.step_size
            out_grid += grid

        # mask out dead cells
        alive_mask_pre = self.grid_alive(grid)
        alive_mask_post = self.grid_alive(out_grid)
        out_grid = out_grid * alive_mask_pre# * alive_mask_post

        return out_grid
    
    def grid_alive(self, grid):
        alive = grid[:, self.alpha_channel_idx, :, :]
        alive_mask = max_pool2d(alive.unsqueeze(1),
                                kernel_size=3, stride=1,
                                padding=1) > 0.1
        return alive_mask.float()

    
    def get_loss(self, grid, target):
        loss = mse_loss(grid[:, :self.visible_channels, :, :],
                        target.repeat(grid.shape[0], 1, 1, 1))
        return loss


if __name__ == "__main__":
    import torchvision
    import numpy as np
    from tqdm import tqdm
    DIM = 16
    H = 32
    W = 32
    CONVOLUTIONAL = True
    BATCH_SIZE = 64

    N_EPOCHS = 200
    BPTT_STEPS = 90
    TEST_STEPS = 180

    COL_DIM = FULL_FILTER.shape[0] * DIM
    FULL_FILTER = FULL_FILTER.repeat(DIM, DIM, 1, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FULL_FILTER = FULL_FILTER.to(device)

    pikachu = torchvision.io.read_image("figures/pikachu.jpg").to(torch.float32)
    pikachu = pikachu.to(device).unsqueeze(0)
    # invert colors
    pikachu = (255 - pikachu) / 255

    pikachu = interpolate(pikachu, (H, W))

    # add alpha channel to pikachu
    alpha_channel = (pikachu.sum(dim=1, keepdim=True) != 0).float()
    pikachu = torch.cat([pikachu, alpha_channel], dim=1)
   
    nca = FixedKernelNCA(DIM, FULL_FILTER,
                         convolutional = CONVOLUTIONAL,
                         h = H, w = W)
    nca = nca.to(device)
    optimizer = torch.optim.Adam(nca.parameters(),
                                 lr = 1e-4)
    pbar = tqdm(range(N_EPOCHS * BPTT_STEPS))
    losses = []
    for epoch in range(N_EPOCHS):
        grid = torch.zeros(BATCH_SIZE, DIM, H, W,
                           device = device)
        # single pixel in the center
        grid[:, :, H // 2, W // 2] = 1
        loss_step = np.random.randint(BPTT_STEPS // 2,
                                      BPTT_STEPS)
        for bptt_step in range(BPTT_STEPS):
            grid = nca(grid)
            if bptt_step == loss_step:
                pbar.update(BPTT_STEPS - bptt_step)
                break
            pbar.update(1)

        optimizer.zero_grad()
        loss = nca.get_loss(grid, pikachu)
        loss.backward()
        # normalize gradients
        torch.nn.utils.clip_grad_norm_(nca.parameters(), 1.0)
        optimizer.step()
        grid = grid.clone().detach()
        pbar.set_description(f"L{loss.item():.4f}")
        losses.append(loss.item())
    pbar.close()
    with torch.no_grad():
        frames = []
        grid = torch.zeros(1, DIM, H, W,
                           device = device)
        grid[:, :, H // 2, W // 2] = 1
        for step in tqdm(range(TEST_STEPS)):
            grid = nca(grid, stochastic = False)
            frame = grid[0, :3, : ,:]
            # back convert
            frame = 255 - 255 * frame
            frame = frame.transpose(0, -1).clip(0, 255).to(torch.int32)
            frames.append(frame.detach().cpu().numpy())
        plot_stack(frames)
            
