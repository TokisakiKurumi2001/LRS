from transformers.configuration_utils import PretrainedConfig

class LRSConfig(PretrainedConfig):
    model_type = "lrs"

    def __init__(
        self,
        input_dim=6,
        hidden_size=10,
        output_dim=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim