import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch.optim import AdamW
from src.data_pipeline import dataloader
from src.models.semantic_memory import SemanticMemory
from src.models.episodic_memory import EpisodicMemory, ManifoldEncoder
from src.models.decoder import Decoder
from jax import random
import jax
import jax.numpy as jnp
import random as py_random
from collections import deque
import math
import optax
from flax.training import train_state
import numpy as np

# --- Hyperparameters for Consolidation ---
LEARNING_RATE = 5e-5
REPLAY_BUFFER_SIZE = 100
SLEEP_PHASE_STEPS = 5
MANIFOLD_ALIGNMENT_LAMBDA = 0.5
GENERATIVE_REPLAY_RATIO = 0.5
BLOCK_SIZE = 128
BATCH_SIZE = 4
DECODER_LR = 1e-4

# --- Hyperparameters for Meta-Controller ---
UNCERTAINTY_THRESHOLD = 0.8
NOVELTY_SATURATION_THRESHOLD = 0.5
INTERFERENCE_RISK_THRESHOLD = 0.9
NOVELTY_WINDOW = 5


# --- JAX Optimizer State ---
# We'll use Optax for a JAX-compatible optimizer.
class JAXTrainState(train_state.TrainState):
    pass


# --- SumTree and Prioritized Replay Buffer for efficient sampling ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.data_pointer = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.episode_embeddings = {}
        self.size = 0  # Track number of non-None entries

    def add(self, episode, priority, episode_embedding):
        # Store current pointer and old data BEFORE tree.add
        index = self.tree.data_pointer
        old_data = self.tree.data[index]

        self.tree.add(priority, episode)

        # Update size: increment only if replacing None
        if old_data is None:
            if self.size < self.capacity:
                self.size += 1

        # Store embedding at original index
        self.episode_embeddings[index] = episode_embedding

    def sample(self, batch_size):
        sampled_data = []
        if self.size == 0:  # Check if buffer is empty
            return sampled_data

        # Sample until we get batch_size non-None entries
        while len(sampled_data) < batch_size:
            segment = self.tree.total_priority / batch_size
            v = py_random.uniform(0, self.tree.total_priority)
            leaf_idx, priority, data = self.tree.get_leaf(v)
            if data is not None:
                sampled_data.append(data)
        return sampled_data

    def get_all_embeddings(self):
        embeddings = []
        for i in range(self.capacity):
            if self.tree.data[i] is not None and i in self.episode_embeddings:
                embeddings.append(self.episode_embeddings[i])
        return jnp.array(embeddings) if embeddings else jnp.array([])


# --- Function to calculate priority ---
def calculate_priority(episode_embedding, replay_buffer):
    if replay_buffer.size > 0:  # Use size instead of len(data)
        all_embeddings = replay_buffer.get_all_embeddings()
        avg_embedding = jnp.mean(all_embeddings, axis=0)
        novelty = jnp.linalg.norm(episode_embedding - avg_embedding)
    else:
        novelty = 1.0
    salience = jnp.linalg.norm(episode_embedding)
    priority = (novelty * 0.5) + (salience * 0.5)
    return priority, novelty


# --- Meta-Controller for adaptive scheduling ---
class MetaController:
    def __init__(self, batch_size):
        self.novelty_scores = deque(maxlen=NOVELTY_WINDOW)
        self.is_sleeping = False
        self.batch_size = batch_size

    def should_sleep(self, semantic_outputs, current_novelty, semantic_embeddings, episodic_embedding):
        logits = semantic_outputs.logits
        probabilities = F.softmax(torch.tensor(logits), dim=-1)
        entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1).mean().item()

        self.novelty_scores.append(current_novelty)
        rolling_avg_novelty = sum(self.novelty_scores) / len(self.novelty_scores)

        semantic_avg_embedding = torch.mean(semantic_embeddings, dim=0)
        episodic_embedding_pt = torch.tensor(episodic_embedding, dtype=torch.float32)

        interference_risk = F.cosine_similarity(semantic_avg_embedding.unsqueeze(0),
                                                episodic_embedding_pt.unsqueeze(0)).item()

        print(
            f"  Metrics: Uncertainty={entropy:.2f}, Novelty_Avg={rolling_avg_novelty:.2f}, Interference_Risk={interference_risk:.2f}")

        if entropy > UNCERTAINTY_THRESHOLD or \
                rolling_avg_novelty > NOVELTY_SATURATION_THRESHOLD or \
                interference_risk > INTERFERENCE_RISK_THRESHOLD:
            self.is_sleeping = True
            return True

        self.is_sleeping = False
        return False


# --- Validation Function ---
def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return perplexity


def train_model(semantic_model, episodic_memory, manifold_encoder, decoder, train_dataloader, val_dataloader,
                optimizer_pt, num_epochs, device, batch_size):

    meta_controller = MetaController(batch_size=batch_size)
    replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)
    key = random.PRNGKey(0)

    # Initialize JAX models and optimizers
    dummy_episode = jnp.zeros((BLOCK_SIZE,), dtype=jnp.int32)

    # Initialize episodic memory
    episodic_params = episodic_memory.init(key, dummy_episode)['params']
    episode_embedding = episodic_memory.apply({'params': episodic_params}, dummy_episode)

    # Initialize manifold encoder
    manifold_params = manifold_encoder.init(key, episode_embedding)['params']
    manifold_embedding = manifold_encoder.apply({'params': manifold_params}, episode_embedding)

    # Prepare batched input for decoder initialization
    decoder_input = jnp.expand_dims(manifold_embedding, axis=0)  # Add batch dimension
    decoder_params = decoder.init(key, decoder_input)['params']

    optimizer_jax = optax.adam(learning_rate=DECODER_LR)
    jax_optimizer_state = optimizer_jax.init(decoder_params)

    @partial(jax.jit, static_argnames=('decoder',))
    def update_decoder(params, opt_state, manifold_embeddings, original_ids, decoder):
        def loss_fn(params, manifold_embeddings, original_ids):
            reconstructed_logits = decoder.apply({'params': params}, manifold_embeddings)
            one_hot_labels = jax.nn.one_hot(original_ids, num_classes=decoder.vocab_size)
            loss = optax.softmax_cross_entropy(logits=reconstructed_logits, labels=one_hot_labels)
            return jnp.mean(loss)

        loss, grads = jax.value_and_grad(loss_fn)(params, manifold_embeddings, original_ids)
        updates, opt_state = optimizer_jax.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # projection layer to align semantic and manifold embeddings
    projection_layer = nn.Linear(768, 64).to(device)
    optimizer_pt = AdamW(
        list(semantic_model.parameters()) + list(projection_layer.parameters()),
        lr=LEARNING_RATE
    )

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        for i, batch in enumerate(train_dataloader):
            semantic_model.train()
            # --- Wake Phase ---
            if not meta_controller.is_sleeping:
                print(f"\n--- Wake Phase: Batch {i + 1} ---")

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                jax_input_ids = jnp.array(input_ids.cpu().numpy())

                with torch.no_grad():
                    semantic_outputs = semantic_model(input_ids=input_ids, attention_mask=attention_mask)
                    semantic_embeddings_pt = semantic_model.distilbert(input_ids=input_ids).last_hidden_state
                    semantic_embeddings_pt = torch.mean(semantic_embeddings_pt, dim=1)

                # Process each episode in the batch
                for episode in jax_input_ids:
                    episode_embedding = episodic_memory.apply({'params': episodic_params}, episode)
                    priority, novelty = calculate_priority(episode_embedding, replay_buffer)

                    # Convert to PyTorch tensor for cosine similarity calculation
                    episode_embedding_pt = torch.tensor(np.array(episode_embedding), dtype=torch.float32)

                    meta_controller.should_sleep(
                        semantic_outputs,
                        novelty,
                        semantic_embeddings_pt,
                        episode_embedding_pt
                    )

                    manifold_embedding = manifold_encoder.apply({'params': manifold_params}, episode_embedding)
                    episode_data = {
                        'input_ids': episode,
                        'manifold_embedding': manifold_embedding,
                    }
                    replay_buffer.add(episode_data, priority, episode_embedding)

            # --- Sleep Phase ---
            else:
                print("\n--- Sleep Phase: Consolidating Knowledge ---")

                # Skip if buffer doesn't have enough data
                if replay_buffer.size < batch_size:
                    print(f"  Only {replay_buffer.size} episodes in buffer. Skipping sleep.")
                    meta_controller.is_sleeping = False
                    continue

                optimizer_pt.zero_grad()
                total_loss = 0.0

                for step in range(SLEEP_PHASE_STEPS):
                    sampled_episodes = replay_buffer.sample(batch_size=batch_size)
                    if not sampled_episodes:
                        continue

                    # Generative Replay Loop
                    num_generative = int(batch_size * GENERATIVE_REPLAY_RATIO)
                    if num_generative > 0 and len(sampled_episodes) > num_generative:
                        manifold_embeddings = jnp.array(
                            [ep['manifold_embedding'] for ep in sampled_episodes[:num_generative]])
                        original_ids = jnp.array([ep['input_ids'] for ep in sampled_episodes[:num_generative]])

                        decoder_params, jax_optimizer_state, gen_loss_val = update_decoder(
                            decoder_params,
                            jax_optimizer_state,
                            manifold_embeddings,
                            original_ids,
                            decoder  # Pass decoder as argument
                        )

                        total_loss += torch.tensor(gen_loss_val.tolist(), device=device)

                        sampled_episodes = sampled_episodes[num_generative:]

                    # Consolidation Loop
                    if sampled_episodes:
                        input_ids_tensor = torch.stack(
                            [torch.tensor(ep['input_ids'].tolist()) for ep in sampled_episodes]).to(device)

                        # Forward pass through semantic model
                        semantic_outputs = semantic_model(input_ids=input_ids_tensor, labels=input_ids_tensor)
                        lm_loss = semantic_outputs.loss

                        # Get semantic embeddings
                        semantic_embeddings_pt = semantic_model.distilbert(input_ids_tensor).last_hidden_state
                        semantic_embeddings_pt = torch.mean(semantic_embeddings_pt, dim=1)

                        # Get stored manifold embeddings
                        consolidated_embeddings = torch.tensor(
                            np.array([ep['manifold_embedding'] for ep in sampled_episodes]),
                            device=device
                        )

                        # Project semantic embeddings to manifold space
                        projected_semantic = projection_layer(semantic_embeddings_pt)

                        # Calculate alignment loss
                        alignment_loss = F.mse_loss(projected_semantic, consolidated_embeddings)

                        total_loss += lm_loss + MANIFOLD_ALIGNMENT_LAMBDA * alignment_loss

                if total_loss > 0:
                    total_loss.backward()
                    optimizer_pt.step()

                print(f"  Consolidation complete. Final Loss: {total_loss.item():.4f}")
                meta_controller.is_sleeping = False

        # Validation step
        if (i + 1) % 10 == 0:
            perplexity = validate_model(semantic_model, val_dataloader, device)
            print(f"  Validation Perplexity after batch {i + 1}: {perplexity:.2f}")


def main():
    print("Starting MEC-Net++ application...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    semantic_memory = SemanticMemory()
    semantic_model = semantic_memory.get_model().to(device)

    episodic_memory = EpisodicMemory()
    manifold_encoder = ManifoldEncoder()
    decoder = Decoder(vocab_size=semantic_memory.get_config().vocab_size, block_size=BLOCK_SIZE)
    print("All models instantiated.")

    train_dataloader = dataloader
    val_dataloader = dataloader

    # Initialize optimizer later to include projection layer
    train_model(semantic_model, episodic_memory, manifold_encoder, decoder, train_dataloader, val_dataloader,
                None, num_epochs=3, device=device, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()
#
# import torch
# import torch.nn as nn
# from functools import partial
# import torch.nn.functional as F
# from torch.optim import AdamW
# from src.data_pipeline import dataloader
# from src.models.semantic_memory import SemanticMemory
# from src.models.episodic_memory import EpisodicMemory, ManifoldEncoder
# from src.models.decoder import Decoder
# from jax import random
# import jax
# import jax.numpy as jnp
# import random as py_random
# from collections import deque
# import math
# import optax
# from flax.training import train_state
# import numpy as np
# import os
# import shutil
# from flax import serialization
# import argparse
#
# # --- Hyperparameters for Consolidation ---
# LEARNING_RATE = 5e-5
# REPLAY_BUFFER_SIZE = 100
# SLEEP_PHASE_STEPS = 5
# MANIFOLD_ALIGNMENT_LAMBDA = 0.5
# GENERATIVE_REPLAY_RATIO = 0.5
# BLOCK_SIZE = 128
# BATCH_SIZE = 4
# DECODER_LR = 1e-4
# CHECKPOINT_DIR = "./checkpoints"
# CHECKPOINT_FREQ = 100  # Save every 100 batches
#
# # --- Hyperparameters for Meta-Controller ---
# UNCERTAINTY_THRESHOLD = 0.8
# NOVELTY_SATURATION_THRESHOLD = 0.5
# INTERFERENCE_RISK_THRESHOLD = 0.9
# NOVELTY_WINDOW = 5
#
# # Create checkpoint directory
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#
#
# # --- JAX Optimizer State ---
# class JAXTrainState(train_state.TrainState):
#     pass
#
#
# # --- SumTree and Prioritized Replay Buffer ---
# class SumTree:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = [0] * (2 * capacity - 1)
#         self.data = [None] * capacity
#         self.data_pointer = 0
#
#     def add(self, priority, data):
#         tree_idx = self.data_pointer + self.capacity - 1
#         self.data[self.data_pointer] = data
#         self.update(tree_idx, priority)
#         self.data_pointer = (self.data_pointer + 1) % self.capacity
#
#     def update(self, tree_idx, priority):
#         change = priority - self.tree[tree_idx]
#         self.tree[tree_idx] = priority
#         while tree_idx != 0:
#             tree_idx = (tree_idx - 1) // 2
#             self.tree[tree_idx] += change
#
#     def get_leaf(self, v):
#         parent_idx = 0
#         while True:
#             left_child_idx = 2 * parent_idx + 1
#             right_child_idx = left_child_idx + 1
#             if left_child_idx >= len(self.tree):
#                 leaf_idx = parent_idx
#                 break
#             if v <= self.tree[left_child_idx]:
#                 parent_idx = left_child_idx
#             else:
#                 v -= self.tree[left_child_idx]
#                 parent_idx = right_child_idx
#         data_idx = leaf_idx - self.capacity + 1
#         return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
#
#     @property
#     def total_priority(self):
#         return self.tree[0]
#
#
# class PrioritizedReplayBuffer:
#     def __init__(self, capacity):
#         self.tree = SumTree(capacity)
#         self.capacity = capacity
#         self.episode_embeddings = {}
#         self.size = 0  # Track number of non-None entries
#
#     def add(self, episode, priority, episode_embedding):
#         # Store current pointer and old data BEFORE tree.add
#         index = self.tree.data_pointer
#         old_data = self.tree.data[index]
#
#         self.tree.add(priority, episode)
#
#         # Update size: increment only if replacing None
#         if old_data is None:
#             if self.size < self.capacity:
#                 self.size += 1
#
#         # Store embedding at original index
#         self.episode_embeddings[index] = episode_embedding
#
#     def sample(self, batch_size):
#         sampled_data = []
#         if self.size == 0:  # Check if buffer is empty
#             return sampled_data
#
#         # Sample until we get batch_size non-None entries
#         while len(sampled_data) < batch_size:
#             segment = self.tree.total_priority / batch_size
#             v = py_random.uniform(0, self.tree.total_priority)
#             leaf_idx, priority, data = self.tree.get_leaf(v)
#             if data is not None:
#                 sampled_data.append(data)
#         return sampled_data
#
#     def get_all_embeddings(self):
#         embeddings = []
#         for i in range(self.capacity):
#             if self.tree.data[i] is not None and i in self.episode_embeddings:
#                 embeddings.append(self.episode_embeddings[i])
#         return jnp.array(embeddings) if embeddings else jnp.array([])
#
#
# # --- Function to calculate priority ---
# def calculate_priority(episode_embedding, replay_buffer):
#     if replay_buffer.size > 0:  # Use size instead of len(data)
#         all_embeddings = replay_buffer.get_all_embeddings()
#         avg_embedding = jnp.mean(all_embeddings, axis=0)
#         novelty = jnp.linalg.norm(episode_embedding - avg_embedding)
#     else:
#         novelty = 1.0
#     salience = jnp.linalg.norm(episode_embedding)
#     priority = (novelty * 0.5) + (salience * 0.5)
#     return priority, novelty
#
#
# # --- Meta-Controller for adaptive scheduling ---
# class MetaController:
#     def __init__(self, batch_size):
#         self.novelty_scores = deque(maxlen=NOVELTY_WINDOW)
#         self.is_sleeping = False
#         self.batch_size = batch_size
#
#     def should_sleep(self, semantic_outputs, current_novelty, semantic_embeddings, episodic_embedding):
#         logits = semantic_outputs.logits
#         probabilities = F.softmax(torch.tensor(logits), dim=-1)
#         entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1).mean().item()
#
#         self.novelty_scores.append(current_novelty)
#         rolling_avg_novelty = sum(self.novelty_scores) / len(self.novelty_scores)
#
#         semantic_avg_embedding = torch.mean(semantic_embeddings, dim=0)
#         episodic_embedding_pt = torch.tensor(episodic_embedding, dtype=torch.float32)
#
#         interference_risk = F.cosine_similarity(semantic_avg_embedding.unsqueeze(0),
#                                                 episodic_embedding_pt.unsqueeze(0)).item()
#
#         print(
#             f"  Metrics: Uncertainty={entropy:.2f}, Novelty_Avg={rolling_avg_novelty:.2f}, Interference_Risk={interference_risk:.2f}")
#
#         if entropy > UNCERTAINTY_THRESHOLD or \
#                 rolling_avg_novelty > NOVELTY_SATURATION_THRESHOLD or \
#                 interference_risk > INTERFERENCE_RISK_THRESHOLD:
#             self.is_sleeping = True
#             return True
#
#         self.is_sleeping = False
#         return False
#
#
# # --- Validation Function ---
# def validate_model(model, dataloader, device):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch["input_ids"].to(device)
#             labels = input_ids.clone()
#             outputs = model(input_ids=input_ids, labels=labels)
#             total_loss += outputs.loss.item()
#
#     avg_loss = total_loss / len(dataloader)
#     perplexity = math.exp(avg_loss)
#     return perplexity
#
#
# # --- Checkpoint Saving ---
# def save_checkpoint(epoch, batch_idx, global_step, semantic_model, projection_layer, episodic_params,
#                     manifold_params, decoder_params, jax_optimizer_state, replay_buffer, meta_controller,
#                     optimizer_pt, key, checkpoint_dir):
#     # Create unique checkpoint path
#     ckpt_path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch}_batch{batch_idx}_step{global_step}.pth")
#     print(f"\nSaving checkpoint to {ckpt_path}")
#
#     # Convert JAX states to bytes
#     episodic_bytes = serialization.to_bytes(episodic_params)
#     manifold_bytes = serialization.to_bytes(manifold_params)
#     decoder_bytes = serialization.to_bytes(decoder_params)
#     opt_state_bytes = serialization.to_bytes(jax_optimizer_state)
#
#     # Convert replay buffer to saveable format
#     replay_save = {
#         'capacity': replay_buffer.capacity,
#         'tree': replay_buffer.tree.tree,
#         'data': replay_buffer.tree.data,
#         'data_pointer': replay_buffer.tree.data_pointer,
#         'size': replay_buffer.size,
#         'embeddings': replay_buffer.episode_embeddings
#     }
#
#     # Convert key to saveable format
#     key_save = key.tolist()
#
#     # Save meta-controller state
#     meta_state = {
#         'novelty_scores': list(meta_controller.novelty_scores),
#         'is_sleeping': meta_controller.is_sleeping
#     }
#
#     # Create checkpoint dictionary
#     checkpoint = {
#         'epoch': epoch,
#         'batch_idx': batch_idx,
#         'global_step': global_step,
#         'semantic_state': semantic_model.state_dict(),
#         'projection_state': projection_layer.state_dict(),
#         'optimizer_pt_state': optimizer_pt.state_dict(),
#         'episodic_bytes': episodic_bytes,
#         'manifold_bytes': manifold_bytes,
#         'decoder_bytes': decoder_bytes,
#         'opt_state_bytes': opt_state_bytes,
#         'replay_buffer': replay_save,
#         'meta_controller': meta_state,
#         'key': key_save
#     }
#
#     # Save checkpoint
#     torch.save(checkpoint, ckpt_path)
#     print(f"Checkpoint saved successfully at step {global_step}")
#     return ckpt_path
#
#
# # --- Checkpoint Loading ---
# def load_checkpoint(ckpt_path, semantic_model, projection_layer, optimizer_pt, device):
#     print(f"Loading checkpoint from {ckpt_path}")
#     checkpoint = torch.load(ckpt_path, map_location=device)
#
#     # Load PyTorch states
#     semantic_model.load_state_dict(checkpoint['semantic_state'])
#     projection_layer.load_state_dict(checkpoint['projection_state'])
#     if 'optimizer_pt_state' in checkpoint:
#         optimizer_pt.load_state_dict(checkpoint['optimizer_pt_state'])
#
#     # Reconstruct replay buffer
#     replay_buffer = PrioritizedReplayBuffer(checkpoint['replay_buffer']['capacity'])
#     replay_buffer.tree.tree = checkpoint['replay_buffer']['tree']
#     replay_buffer.tree.data = checkpoint['replay_buffer']['data']
#     replay_buffer.tree.data_pointer = checkpoint['replay_buffer']['data_pointer']
#     replay_buffer.size = checkpoint['replay_buffer']['size']
#     replay_buffer.episode_embeddings = checkpoint['replay_buffer']['embeddings']
#
#     # Recreate meta-controller state
#     meta_controller = MetaController(BATCH_SIZE)
#     meta_controller.novelty_scores = deque(checkpoint['meta_controller']['novelty_scores'],
#                                            maxlen=NOVELTY_WINDOW)
#     meta_controller.is_sleeping = checkpoint['meta_controller']['is_sleeping']
#
#     # Convert key back to JAX format
#     key = jnp.array(checkpoint['key'], dtype=jnp.uint32)
#
#     # Deserialize JAX parameters
#     episodic_params = serialization.from_bytes(episodic_params, checkpoint['episodic_bytes'])
#     manifold_params = serialization.from_bytes(manifold_params, checkpoint['manifold_bytes'])
#     decoder_params = serialization.from_bytes(decoder_params, checkpoint['decoder_bytes'])
#     jax_optimizer_state = serialization.from_bytes(jax_optimizer_state, checkpoint['opt_state_bytes'])
#
#     return (
#         checkpoint['epoch'],
#         checkpoint['batch_idx'],
#         checkpoint['global_step'],
#         key,
#         replay_buffer,
#         meta_controller,
#         episodic_params,
#         manifold_params,
#         decoder_params,
#         jax_optimizer_state
#     )
#
#
# def train_model(semantic_model, episodic_memory, manifold_encoder, decoder, train_dataloader, val_dataloader,
#                 optimizer_pt, num_epochs, device, batch_size, resume_checkpoint=None):
#     # Initialize training state
#     start_epoch = 0
#     start_batch = 0
#     global_step = 0
#     key = random.PRNGKey(0)
#
#     # Initialize components
#     projection_layer = nn.Linear(768, 64).to(device)
#     replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)
#     meta_controller = MetaController(batch_size=batch_size)
#
#     # Initialize JAX models
#     dummy_episode = jnp.zeros((BLOCK_SIZE,), dtype=jnp.int32)
#     episodic_params = episodic_memory.init(key, dummy_episode)['params']
#     manifold_params = manifold_encoder.init(key, episodic_memory.apply({'params': episodic_params}, dummy_episode))[
#         'params']
#     decoder_params = decoder.init(key, jnp.zeros((1, 64)))['params']
#
#     # Initialize JAX optimizer
#     optimizer_jax = optax.adam(learning_rate=DECODER_LR)
#     jax_optimizer_state = optimizer_jax.init(decoder_params)
#
#     # Resume from checkpoint if provided
#     if resume_checkpoint:
#         (start_epoch,
#          start_batch,
#          global_step,
#          key,
#          replay_buffer,
#          meta_controller,
#          episodic_params,
#          manifold_params,
#          decoder_params,
#          jax_optimizer_state) = load_checkpoint(
#             resume_checkpoint,
#             semantic_model,
#             projection_layer,
#             optimizer_pt,
#             device
#         )
#         print(f"Resuming training from epoch {start_epoch + 1}, batch {start_batch + 1}, step {global_step}")
#
#     # Update function for JAX decoder
#     @partial(jax.jit, static_argnames=('decoder',))
#     def update_decoder(params, opt_state, manifold_embeddings, original_ids, decoder):
#         def loss_fn(params, manifold_embeddings, original_ids):
#             reconstructed_logits = decoder.apply({'params': params}, manifold_embeddings)
#             one_hot_labels = jax.nn.one_hot(original_ids, num_classes=decoder.vocab_size)
#             loss = optax.softmax_cross_entropy(logits=reconstructed_logits, labels=one_hot_labels)
#             return jnp.mean(loss)
#
#         loss, grads = jax.value_and_grad(loss_fn)(params, manifold_embeddings, original_ids)
#         updates, opt_state = optimizer_jax.update(grads, opt_state, params)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state, loss
#
#     # Training loop
#     for epoch in range(start_epoch, num_epochs):
#         print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
#
#         # Skip batches if resuming
#         skip_batches = start_batch if epoch == start_epoch else 0
#
#         for i, batch in enumerate(train_dataloader):
#             if i < skip_batches:
#                 continue
#
#             semantic_model.train()
#             # --- Wake Phase ---
#             if not meta_controller.is_sleeping:
#                 print(f"\n--- Wake Phase: Batch {i + 1} ---")
#
#                 input_ids = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 jax_input_ids = jnp.array(input_ids.cpu().numpy())
#
#                 with torch.no_grad():
#                     semantic_outputs = semantic_model(input_ids=input_ids, attention_mask=attention_mask)
#                     semantic_embeddings_pt = semantic_model.distilbert(input_ids=input_ids).last_hidden_state
#                     semantic_embeddings_pt = torch.mean(semantic_embeddings_pt, dim=1)
#
#                 # Process each episode in the batch
#                 for episode in jax_input_ids:
#                     episode_embedding = episodic_memory.apply({'params': episodic_params}, episode)
#                     priority, novelty = calculate_priority(episode_embedding, replay_buffer)
#
#                     # Convert to PyTorch tensor for cosine similarity calculation
#                     episode_embedding_pt = torch.tensor(np.array(episode_embedding), dtype=torch.float32)
#
#                     meta_controller.should_sleep(
#                         semantic_outputs,
#                         novelty,
#                         semantic_embeddings_pt,
#                         episode_embedding_pt
#                     )
#
#                     manifold_embedding = manifold_encoder.apply({'params': manifold_params}, episode_embedding)
#                     episode_data = {
#                         'input_ids': episode,
#                         'manifold_embedding': manifold_embedding,
#                     }
#                     replay_buffer.add(episode_data, priority, episode_embedding)
#
#             # --- Sleep Phase ---
#             else:
#                 print("\n--- Sleep Phase: Consolidating Knowledge ---")
#
#                 # Skip if buffer doesn't have enough data
#                 if replay_buffer.size < batch_size:
#                     print(f"  Only {replay_buffer.size} episodes in buffer. Skipping sleep.")
#                     meta_controller.is_sleeping = False
#                     continue
#
#                 optimizer_pt.zero_grad()
#                 total_loss = 0.0
#
#                 for step in range(SLEEP_PHASE_STEPS):
#                     sampled_episodes = replay_buffer.sample(batch_size=batch_size)
#                     if not sampled_episodes:
#                         continue
#
#                     # Generative Replay Loop
#                     num_generative = int(batch_size * GENERATIVE_REPLAY_RATIO)
#                     if num_generative > 0 and len(sampled_episodes) > num_generative:
#                         manifold_embeddings = jnp.array(
#                             [ep['manifold_embedding'] for ep in sampled_episodes[:num_generative]])
#                         original_ids = jnp.array([ep['input_ids'] for ep in sampled_episodes[:num_generative]])
#
#                         decoder_params, jax_optimizer_state, gen_loss_val = update_decoder(
#                             decoder_params,
#                             jax_optimizer_state,
#                             manifold_embeddings,
#                             original_ids,
#                             decoder
#                         )
#
#                         total_loss += torch.tensor(gen_loss_val.tolist(), device=device)
#
#                         sampled_episodes = sampled_episodes[num_generative:]
#
#                     # Consolidation Loop
#                     if sampled_episodes:
#                         input_ids_tensor = torch.stack(
#                             [torch.tensor(ep['input_ids'].tolist()) for ep in sampled_episodes]).to(device)
#
#                         # Forward pass through semantic model
#                         semantic_outputs = semantic_model(input_ids=input_ids_tensor, labels=input_ids_tensor)
#                         lm_loss = semantic_outputs.loss
#
#                         # Get semantic embeddings
#                         semantic_embeddings_pt = semantic_model.distilbert(input_ids_tensor).last_hidden_state
#                         semantic_embeddings_pt = torch.mean(semantic_embeddings_pt, dim=1)
#
#                         # Get stored manifold embeddings
#                         consolidated_embeddings = torch.tensor(
#                             np.array([ep['manifold_embedding'] for ep in sampled_episodes]),
#                             device=device
#                         )
#
#                         # Project semantic embeddings to manifold space
#                         projected_semantic = projection_layer(semantic_embeddings_pt)
#
#                         # Calculate alignment loss
#                         alignment_loss = F.mse_loss(projected_semantic, consolidated_embeddings)
#
#                         total_loss += lm_loss + MANIFOLD_ALIGNMENT_LAMBDA * alignment_loss
#
#                 if total_loss > 0:
#                     total_loss.backward()
#                     optimizer_pt.step()
#
#                 print(f"  Consolidation complete. Final Loss: {total_loss.item():.4f}")
#                 meta_controller.is_sleeping = False
#
#             # Save checkpoint
#             global_step += 1
#             if global_step % CHECKPOINT_FREQ == 0:
#                 ckpt_path = save_checkpoint(
#                     epoch, i, global_step,
#                     semantic_model, projection_layer,
#                     episodic_params, manifold_params, decoder_params,
#                     jax_optimizer_state,
#                     replay_buffer, meta_controller,
#                     optimizer_pt, key,
#                     CHECKPOINT_DIR
#                 )
#
#                 # Keep only the latest 3 checkpoints
#                 checkpoints = sorted(
#                     [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt_")],
#                     key=lambda x: int(x.split("_step")[1].split(".")[0]),
#                     reverse=True
#                 )
#                 for old_ckpt in checkpoints[3:]:
#                     os.remove(os.path.join(CHECKPOINT_DIR, old_ckpt))
#
#             # Validation step
#             if (global_step) % 10 == 0:
#                 perplexity = validate_model(semantic_model, val_dataloader, device)
#                 print(f"  Validation Perplexity after step {global_step}: {perplexity:.2f}")
#
#         # Reset batch counter after each epoch
#         start_batch = 0
#
#         # Save at end of epoch
#         ckpt_path = save_checkpoint(
#             epoch, len(train_dataloader) - 1, global_step,
#             semantic_model, projection_layer,
#             episodic_params, manifold_params, decoder_params,
#             jax_optimizer_state,
#             replay_buffer, meta_controller,
#             optimizer_pt, key,
#             CHECKPOINT_DIR
#         )
#
#     # Final save after training completes
#     save_checkpoint(
#         num_epochs - 1, len(train_dataloader) - 1, global_step,
#         semantic_model, projection_layer,
#         episodic_params, manifold_params, decoder_params,
#         jax_optimizer_state,
#         replay_buffer, meta_controller,
#         optimizer_pt, key,
#         CHECKPOINT_DIR
#     )
#
#
# def main(resume=False):
#     print("Starting MEC-Net++ application...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     semantic_memory = SemanticMemory()
#     semantic_model = semantic_memory.get_model().to(device)
#     episodic_memory = EpisodicMemory()
#     manifold_encoder = ManifoldEncoder()
#     decoder = Decoder(vocab_size=semantic_memory.get_config().vocab_size, block_size=BLOCK_SIZE)
#     print("All models instantiated.")
#
#     # Initialize optimizer
#     projection_layer = nn.Linear(768, 64).to(device)
#     optimizer_pt = AdamW(
#         list(semantic_model.parameters()) + list(projection_layer.parameters()),
#         lr=LEARNING_RATE
#     )
#
#     # Handle resume
#     resume_checkpoint = None
#     if resume:
#         checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt_")]
#         if checkpoints:
#             # Find checkpoint with highest step number
#             latest_ckpt = max(
#                 checkpoints,
#                 key=lambda x: int(x.split("_step")[1].split(".")[0]))
#             resume_checkpoint = os.path.join(CHECKPOINT_DIR, latest_ckpt)
#             print(f"Found checkpoint: {resume_checkpoint}")
#         else:
#             print("No checkpoints found, starting from scratch")
#
#             train_dataloader = dataloader
#             val_dataloader = dataloader
#
#             # Start training
#             train_model(
#                 semantic_model=semantic_model,
#                 episodic_memory=episodic_memory,
#                 manifold_encoder=manifold_encoder,
#                 decoder=decoder,
#                 train_dataloader=train_dataloader,
#                 val_dataloader=val_dataloader,
#                 optimizer_pt=optimizer_pt,
#                 num_epochs=3,
#                 device=device,
#                 batch_size=BATCH_SIZE,
#                 resume_checkpoint=resume_checkpoint
#             )
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
#     args = parser.parse_args()
#     main(resume=args.resume)