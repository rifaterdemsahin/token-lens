"""Registry of supported models. slug -> Hugging Face repo + name of the embedding tensor."""
MODELS = {
    "gpt2": {"hf": "openai-community/gpt2", "tensor": "wte.weight", "label": "GPT-2 small (50k vocab, 768d)"},
    "qwen2.5-0.5b": {"hf": "Qwen/Qwen2.5-0.5B", "tensor": "model.embed_tokens.weight",
                     "label": "Qwen2.5-0.5B (152k vocab, 896d, multilingual)"},
    "gemma-3-270m": {"hf": "unsloth/gemma-3-270m", "tensor": "model.embed_tokens.weight",
                     "label": "Gemma-3-270m (262k vocab, 640d, multilingual)"},
    "bert-turkish": {"hf": "dbmdz/bert-base-turkish-cased", "tensor": "bert.embeddings.word_embeddings.weight",
                     "tokenizer": "wordpiece", "lang": "tr",
                     "label": "BERT Turkish cased (32k Turkish-only vocab, 768d) - most Turkish-friendly"},
}
DEFAULT_MODEL = "gpt2"
