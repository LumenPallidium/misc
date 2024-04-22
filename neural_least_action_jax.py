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
        self.out_indices = jnp.arange(self.full_dim - self.out_dim, self.full_dim)
        # note output neurons are part of the recurrent group
        self.recurrent_indices = jnp.arange(self.in_dim, self.full_dim)

        # TODO: can revist .params later
        self.W = jax.random.normal(jax.random.PRNGKey(0),
                                   (self.full_dim, self.dim))
        self.voltage = jnp.zeros((self.batch_size, self.dim))

    def __call__(self,
                 x,
                 expected_output: jnp.ndarray = None):
        rate = double_relu(self.voltage)
        x = jnp.concatenate([x, rate], axis=-1)

        activation = x @ self.W
        voltage_error = self.voltage - activation
        dW = self.lr * jnp.einsum('bi, bj -> bij', voltage_error, x)

        output_error = jnp.zeros((self.batch_size,
                                  self.full_dim))
        if expected_output is not None:
            output_error = output_error.at[self.out_indices].set(expected_output - self.voltage[self.out_indices])

        # TODO : feel like there should be a better way to do this
        rate_grad = jax.jvp(double_relu,
                            (self.voltage, ),
                            (jnp.ones_like(self.voltage), ))[1]
        W_net = self.W[self.recurrent_indices, self.recurrent_indices]
        back_error = rate_grad * (W_net.T @ voltage_error) + self.output_nudge * output_error

        # TODO: paper algorithm 1 says this requires some time derivatives,
        # but this formualation matches "Details for Fig. 4" section
        dv = -self.voltage + activation + back_error
        dv = dv / self.time_constant

        self.voltage += self.step_size * dv
        self.W += self.step_size * dW
        return self.voltage

batch_size = 32
model = MultiCompartmental(dim = 10,
                           in_dim = 5,
                           batch_size = batch_size)
batch = jnp.ones((batch_size, 5))
variables = model.init(jax.random.key(0), batch)
output = model.apply(variables, batch)