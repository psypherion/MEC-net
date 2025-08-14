from transformers import DistilBertForMaskedLM, DistilBertConfig


class SemanticMemory:
    """
    The Semantic Memory model, representing the long-term, generalized knowledge.
    This is a small Transformer-based language model.
    """

    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initializes the SemanticMemory model.

        Args:
            model_name (str): The name of the pre-trained model to use.
        """
        # Load a pre-trained model and configuration.
        # We use DistilBert because it's smaller and more efficient for our hardware.
        try:
            self.model = DistilBertForMaskedLM.from_pretrained(model_name)
            self.config = DistilBertConfig.from_pretrained(model_name)
            print(f"Semantic Memory model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading Semantic Memory model: {e}")
            print("Creating a model with a default configuration instead.")
            self.config = DistilBertConfig()
            self.model = DistilBertForMaskedLM(self.config)

    def get_model(self):
        """
        Returns the underlying Hugging Face model instance.
        """
        return self.model

    def get_config(self):
        """
        Returns the model's configuration.
        """
        return self.config

    def __call__(self, input_ids, attention_mask):
        """
        Performs a forward pass through the model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.

        Returns:
            torch.Tensor: The model's output (logits).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits


if __name__ == '__main__':
    # This block allows us to test the model independently.
    # It creates a dummy input and runs a forward pass.
    print("Testing SemanticMemory model...")
    semantic_memory = SemanticMemory()

    # Create a dummy batch of input IDs and attention mask
    dummy_input_ids = torch.randint(0, semantic_memory.config.vocab_size, (4, 128))
    dummy_attention_mask = torch.ones(4, 128, dtype=torch.long)

    # Perform a forward pass
    logits = semantic_memory(dummy_input_ids, dummy_attention_mask)

    print(f"Input shape: {dummy_input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print("SemanticMemory test successful.")
