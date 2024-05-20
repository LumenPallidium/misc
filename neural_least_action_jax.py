import jax
import jax.numpy as jnp
import flax.linen as nn

@jax.jit
def double_relu(x, x_max=6.0):
  """
  Function that applies a double ReLU activation function to the input i.e.
  p(x) = x if x > 0 and x < x_max, 0 or x_max otherwise (respectively).
  """
  return jnp.where(x > 0, jnp.where(x > x_max, x_max, x), 0)


class MultiCompartmental(nn.Module):
    """
    The multicompartmental neuron from the Neural Least Action principle paper.
    """
    dim : int
    in_dim : int
    batch_size : int = 32
    out_dim : int = 1
    lr : float = 1e-3
    step_size : float = 1e-3
    output_nudge : float = 0.1
    time_constant : float = 1e-2

    def setup(self):
        self.full_dim = self.dim + self.in_dim
        self.out_indices = jnp.arange(self.dim - self.out_dim, self.dim)
        # note output neurons are part of the recurrent group
        self.recurrent_indices = jnp.arange(self.in_dim, self.full_dim)

    @nn.compact
    def __call__(self,
                 x,
                 expected_output: jnp.ndarray = None):
        
        W = self.variable('W', 'W',
                          nn.initializers.normal(),
                          jax.random.PRNGKey(0),
                          (self.batch_size, self.full_dim, self.dim))
        voltage = self.variable('voltage',
                                'voltage',
                                jnp.zeros,
                                (self.batch_size, self.dim))

        rate = double_relu(voltage.value)
        x = jnp.concatenate([x, rate], axis=-1)

        activation = jnp.einsum("bji,bj->bi", W.value, x) #x @ W.value
        voltage_error = voltage.value - activation
        dW = self.lr * jnp.einsum("bi, bj -> bij", voltage_error, x)

        output_error = jnp.zeros((self.batch_size,
                                  self.dim))
        if expected_output is not None:
            output_error = output_error.at[:, self.out_indices].set(expected_output - voltage.value[:, self.out_indices])

        # TODO : feel like there should be a better way to do this
        rate_grad = jax.jvp(double_relu,
                            (voltage.value, ),
                            (jnp.ones_like(voltage.value), ))[1]
        W_net = W.value[:, self.recurrent_indices, :]
        back_error = jnp.einsum("bij,bj->bi", W_net, voltage_error)
        back_error = rate_grad * back_error + self.output_nudge * output_error

        # TODO: paper algorithm 1 says this requires some time derivatives,
        # but this formualation matches "Details for Fig. 4" section
        dv = -voltage.value + activation + back_error
        dv = dv / self.time_constant

        voltage.value += self.step_size * dv
        W.value += self.step_size * dW.mean(axis=0).T
        return voltage.value.copy(), output_error.mean()


def runge_kutta(f, initial_conditions, n_steps, dt = 1e-4):
    """Runge-Kutta 4th order integration of a dynamical system.
    
    Parameters
    ----------
    f : callable
        The dynamical system, a function of time and state i.e. f(t, x).
    initial_conditions : jnp.ndarray
        The initial conditions of the dynamical system.
    n_steps : int
        The number of steps to integrate for.
    dt : float
        The time step.
    """

    y0 = initial_conditions.astype(jnp.float32)
    values = [y0[jnp.newaxis, :]]

    t_i = 0

    for i in range(n_steps):
        y_i = values[i][0]
        dtd2 = 0.5 * dt
        f1 = f(y_i)
        f2 = f(y_i + dtd2 * f1)
        f3 = f(y_i + dtd2 * f2)
        f4 = f(y_i + dt * f3)
        dy = 1/6 * dt * (f1 + 2 * (f2 + f3) +f4)
        y_next = y_i + dy
        y_next = y_next
        t_i += dt

        values.append(y_next[jnp.newaxis, :])

    return jnp.concat(values, axis = 0)


class DynamicalSystemSampler:
    """
    Simple class to generate samples based on a dynamical system,
    which is specified as a function of the state.
    """
    def __init__(self, key, dynamical_system, n_steps, batch_size = 32, dt=10e-5, dim = 4):
        self.key = key
        self.dynamical_system = dynamical_system
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.dt = dt
        self.dim = dim

    def sample(self):
        initial_conditions = jax.random.normal(self.key,
                                               (self.batch_size, self.dim))
        return runge_kutta(self.dynamical_system,
                           initial_conditions,
                           self.n_steps,
                           self.dt)
def lorenz4d(x,
             a : float = 5,
             b : float = 20,
             c : float = 1,
             d : float = 0.1,
             e : float = 0.1,
             f : float = 20.6,
             g : float = 1):
        """
        A 4D variant of the Lorenz system.
        """
        # split into batches of components
        x, y, z, w = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        dxdt = a * (y - x) - f * w
        dydt = x * z - g * y
        dzdt = b - x * y - c * z
        dwdt = d * y - e * w
        return jnp.stack([dxdt, dydt, dzdt, dwdt], axis=-1)

#TODO : errors not going down?
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    total_steps = 1000
    batch_size = 32

    key = jax.random.PRNGKey(0)
    model = MultiCompartmental(dim = 124,
                               in_dim = 4,
                               out_dim = 4,
                               batch_size = batch_size)
    sampler = DynamicalSystemSampler(key,
                                     lorenz4d,
                                     total_steps,
                                     batch_size = batch_size,
                                     dim = 4)
    dataset = sampler.sample()
    batch = dataset[0, :]
    
    errors = []
    for i in tqdm(range(total_steps)):
        expected_output = dataset[i, :]

        variables = model.init(key, batch)
        (voltage, error), variables = model.apply(variables,
                                        batch,
                                        expected_output=expected_output,
                                        mutable=['voltage', 'W'])
        errors.append(error.mean().item())
        batch = voltage[:, -4:].copy()

    plt.plot(errors)
    plt.show()
