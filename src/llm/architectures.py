"""Architecture identifiers shared across CLI, model, and training."""

ARCH_LEGACY = "gpt_learnedpos_layernorm_gelu_v0"
ARCH_MODERN = "gpt_rope_rmsnorm_swiglu_v1"
SUPPORTED_ARCHITECTURES = {ARCH_LEGACY, ARCH_MODERN}

