import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn
from jax import random


class GNNModel(nn.Module):
    """
    A simple Graph Neural Network to process our episodic data.
    """

    @nn.compact
    def __call__(self, graph):
        # The core of the GNN is a single GraphNetwork layer.
        # This layer can perform message passing and update node features.
        gnn = jraph.GraphNetwork(
            update_node_fn=lambda nodes, sent_edges, globals, sent_globals:
            nn.Dense(features=nodes.shape[-1])(nodes),
            update_edge_fn=None,
            update_global_fn=None
        )
        return gnn(graph)


class EpisodicMemory(nn.Module):
    """
    The Episodic Memory model, using a GNN to store short-term, high-plasticity memories.
    """
    hidden_dims: int = 128
    output_dims: int = 768  # Updated to match the Semantic Memory's hidden dimension

    @nn.compact
    def __call__(self, input_ids):
        # Flatten the input_ids to create a list of nodes.
        # A simple embedding layer to convert token IDs into features.
        nodes = nn.Embed(num_embeddings=30522, features=self.hidden_dims)(input_ids)

        # A simplified way to create a sequential graph.
        # Each token is a node, and edges connect adjacent tokens.
        num_nodes = nodes.shape[0]
        senders = jnp.arange(num_nodes - 1)
        receivers = jnp.arange(1, num_nodes)

        graph = jraph.GraphsTuple(
            n_node=jnp.array([num_nodes]),
            n_edge=jnp.array([num_nodes - 1]),
            nodes=nodes,
            edges=jnp.ones((num_nodes - 1, 1)),
            globals=None,
            senders=senders,
            receivers=receivers
        )

        # Process the graph with the GNN model.
        processed_graph = GNNModel()(graph)

        # We'll project the processed nodes into the Semantic Memory's dimension
        # using a linear layer.
        processed_nodes = nn.Dense(features=self.output_dims)(processed_graph.nodes)

        # For a simple prototype, we'll just take the mean of the processed nodes
        # to get a single episode embedding.
        episode_embedding = jnp.mean(processed_nodes, axis=0)

        return episode_embedding


class ManifoldEncoder(nn.Module):
    """
    The Manifold Encoder, projecting high-dimensional episodic data into a
    low-dimensional manifold space.
    """
    manifold_dims: int = 64

    @nn.compact
    def __call__(self, episode_embedding):
        # A simple MLP to encode the embedding into the manifold space.
        x = nn.Dense(features=self.manifold_dims)(episode_embedding)
        x = nn.relu(x)
        x = nn.Dense(features=self.manifold_dims)(x)
        return x


if __name__ == "__main__":
    # This block allows us to test the models independently.
    key = random.PRNGKey(0)

    print("Testing EpisodicMemory and ManifoldEncoder models...")

    # Create a dummy batch of input IDs (PyTorch tensor)
    # The batch size is 4, and block size is 128.
    dummy_input_ids_pt = torch.randint(0, 30522, (4, 128))

    # Convert a single episode to a JAX array for the model
    dummy_episode_jax = jnp.array(dummy_input_ids_pt[0])

    # Initialize the models
    episodic_memory = EpisodicMemory()
    manifold_encoder = ManifoldEncoder()


    # Define a JAX function to run the forward pass
    @jax.jit
    def run_models(params, input_data):
        episode_embedding = episodic_memory.apply(params['episodic_memory'], input_data)
        manifold_embedding = manifold_encoder.apply(params['manifold_encoder'], episode_embedding)
        return episode_embedding, manifold_embedding


    # Initialize parameters for the models
    params = {
        'episodic_memory': episodic_memory.init(key, dummy_episode_jax),
        'manifold_encoder': manifold_encoder.init(key, episodic_memory.init(key, dummy_episode_jax))
    }

    # Run the models with the dummy data
    episode_embedding_out, manifold_embedding_out = run_models(params, dummy_episode_jax)

    print(f"Dummy episode input shape: {dummy_episode_jax.shape}")
    print(f"Episode embedding shape: {episode_embedding_out.shape}")
    print(f"Manifold embedding shape: {manifold_embedding_out.shape}")
    print("EpisodicMemory and ManifoldEncoder test successful.")
