import os
import xml.etree.ElementTree as ET
from typing import List
import torch

# Import NeoMind classes
# from neomind_model import NeoMindConfig, NeoMindModel

def parse_list(text: str) -> List[float]:
    text = text.strip("[] ")
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def load_cfml_config(cfml_path: str):
    tree = ET.parse(cfml_path)
    root = tree.getroot()
    model_node = root.find("model")

    rope_node = model_node.find("rope_scaling")
    rope_scaling = {
        "type": rope_node.find("type").text,
        "short_factor": parse_list(rope_node.find("short_factor").text),
        "long_factor": parse_list(rope_node.find("long_factor").text)
    } if rope_node is not None else None

    return NeoMindConfig(
        vocab_size=int(model_node.find("vocab_size").text),
        hidden_size=int(model_node.find("hidden_size").text),
        intermediate_size=int(model_node.find("intermediate_size").text),
        num_hidden_layers=int(model_node.find("num_hidden_layers").text),
        num_attention_heads=int(model_node.find("num_attention_heads").text),
        num_key_value_heads=int(model_node.find("num_key_value_heads").text),
        resid_pdrop=float(model_node.find("resid_pdrop").text),
        embd_pdrop=float(model_node.find("embd_pdrop").text),
        attention_dropout=float(model_node.find("attention_dropout").text),
        hidden_act=model_node.find("hidden_act").text,
        max_position_embeddings=int(model_node.find("max_position_embeddings").text),
        original_max_position_embeddings=int(model_node.find("original_max_position_embeddings").text),
        use_cache=model_node.find("use_cache").text.lower() == "true",
        tie_word_embeddings=model_node.find("tie_word_embeddings").text.lower() == "true",
        rope_theta=float(model_node.find("rope_theta").text),
        rope_scaling=rope_scaling,
        bos_token_id=int(model_node.find("bos_token_id").text),
        eos_token_id=int(model_node.find("eos_token_id").text),
        pad_token_id=int(model_node.find("pad_token_id").text),
        sliding_window=int(model_node.find("sliding_window").text),
    )

class NeoMindHub:
    def __init__(self, configs_folder: str):
        self.configs_folder = configs_folder
        self.models = {}  # Cache loaded models
        self.available_models = self._scan_configs()
    
    def _scan_configs(self):
        """Scan folder for available CFML configs."""
        models = []
        for f in os.listdir(self.configs_folder):
            if f.endswith(".cfml"):
                models.append(os.path.splitext(f)[0])
        return models

    def list_models(self):
        """List available model names."""
        return self.available_models

    def get_model_info(self, model_name: str):
        """Return basic info without loading full model."""
        cfml_path = os.path.join(self.configs_folder, model_name + ".cfml")
        if not os.path.exists(cfml_path):
            raise ValueError(f"Model {model_name} not found in {self.configs_folder}")
        config = load_cfml_config(cfml_path)
        return {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_layers": config.num_hidden_layers,
            "attention_heads": config.num_attention_heads,
            "sliding_window": config.sliding_window,
            "max_position_embeddings": config.max_position_embeddings
        }

    def load_model(self, model_name: str):
        """Load NeoMind model from CFML. Caches result."""
        if model_name in self.models:
            return self.models[model_name]
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available.")
        cfml_path = os.path.join(self.configs_folder, model_name + ".cfml")
        config = load_cfml_config(cfml_path)
        model = NeoMindModel(config)
        self.models[model_name] = model
        return model
