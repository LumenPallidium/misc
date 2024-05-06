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

    @nn.compact
    def __call__(self,
                 x,
                 expected_output: jnp.ndarray = None):
        
        W = self.variable('W', 'W',
                          nn.initializers.normal(),
                          jax.random.PRNGKey(0),
                          (self.full_dim, self.dim))
        voltage = self.variable('voltage',
                                'voltage',
                                jnp.zeros,
                                (self.batch_size, self.dim))

        rate = double_relu(voltage.value)
        x = jnp.concatenate([x, rate], axis=-1)

        activation = x @ W.value
        voltage_error = voltage.value - activation
        dW = self.lr * jnp.einsum('bi, bj -> bij', voltage_error, x)

        output_error = jnp.zeros((self.batch_size,
                                  self.dim))
        if expected_output is not None:
            output_error = output_error.at[self.out_indices].set(expected_output - voltage.value[self.out_indices])

        # TODO : feel like there should be a better way to do this
        rate_grad = jax.jvp(double_relu,
                            (voltage.value, ),
                            (jnp.ones_like(voltage.value), ))[1]
        W_net = W.value[self.recurrent_indices, :]
        back_error = rate_grad * (voltage_error @ W_net) + self.output_nudge * output_error

        # TODO: paper algorithm 1 says this requires some time derivatives,
        # but this formualation matches "Details for Fig. 4" section
        dv = -voltage.value + activation + back_error
        dv = dv / self.time_constant

        voltage.value += self.step_size * dv
        W.value += self.step_size * dW.mean(axis=0).T
        return voltage.value.copy()

batch_size = 32
model = MultiCompartmental(dim = 10,
                           in_dim = 5,
                           batch_size = batch_size)
batch = jnp.ones((batch_size, 5))
variables = model.init(jax.random.key(0), batch)
output = model.apply(variables, batch, mutable=['voltage', 'W'])