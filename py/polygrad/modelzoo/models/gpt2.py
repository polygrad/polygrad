"""
Canonical Python implementation of GPT-2.
This is the single source of truth for the model architecture, parameter naming, and training behavior.
All other language frontends (JS/C) must conform to this exact layout via the model manifest.
"""
from polygrad.tensor import Tensor

class GPT2Config:
    def __init__(self, vocab_size=50257, max_seq_len=1024, dim=768, n_layers=12, n_heads=12):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads

class GPT2:
    def __init__(self, config: GPT2Config):
        self.config = config
        
        # Initialize parameters exactly matching the manifest naming
        self.wte = Tensor.randn(config.vocab_size, config.dim, requires_grad=True)
        self.wpe = Tensor.randn(config.max_seq_len, config.dim, requires_grad=True)
        
        # Blocks would be initialized here...
        self.blocks = []

    def load_state_dict(self, state_dict: dict):
        """Standardized checkpoint loader ensuring format agnostic parity."""
        self.wte = state_dict['wte.weight']
        self.wpe = state_dict['wpe.weight']

    def get_state_dict(self):
        """Standardized checkpoint saver."""
        return {
            "wte.weight": self.wte,
            "wpe.weight": self.wpe,
            # ...
        }

    def forward(self, input_ids: Tensor) -> Tensor:
        # Standard dynamic forward pass
        # (This builds the UOp graph lazily)
        return input_ids # stub for logits
