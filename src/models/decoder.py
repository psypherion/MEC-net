import jax
import jax.numpy as jnp
from flax import linen as nn


class Decoder(nn.Module):
    """
    A Decoder model to reconstruct text episodes from manifold embeddings.
    This is a simple MLP for our prototype, as a full Transformer decoder
    would be too large for the current hardware constraints.
    """
    vocab_size: int = 30522
    hidden_dims: int = 128
    block_size: int = 128

    @nn.compact
    def __call__(self, manifold_embedding):
        # Ensure we have at least 2 dimensions (batch, features)
        if manifold_embedding.ndim == 1:
            manifold_embedding = jnp.expand_dims(manifold_embedding, axis=0)

        x = nn.Dense(features=self.hidden_dims)(manifold_embedding)
        x = nn.relu(x)
        x = nn.Dense(features=self.block_size * self.vocab_size)(x)

        # Get batch size from input
        batch_size = manifold_embedding.shape[0]
        return x.reshape((batch_size, self.block_size, self.vocab_size))


if __name__ == "__main__":
    # Test the Decoder model
    key = jax.random.PRNGKey(0)
    decoder = Decoder()

    # Create a dummy manifold embedding (e.g., a batch of 1)
    dummy_manifold_embedding = jnp.zeros((1, 64))

    # Initialize the parameters and run a forward pass
    params = decoder.init(key, dummy_manifold_embedding)
    reconstructed_logits = decoder.apply(params, dummy_manifold_embedding)

    print(f"Dummy manifold embedding shape: {dummy_manifold_embedding.shape}")
    print(f"Reconstructed logits shape: {reconstructed_logits.shape}")
    print("Decoder model test successful.")
