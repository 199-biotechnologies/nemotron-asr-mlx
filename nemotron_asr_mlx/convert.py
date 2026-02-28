""".nemo checkpoint to MLX safetensors converter.

Extracts the NeMo tarball, loads the PyTorch checkpoint, remaps weight
names and tensor layouts to match the MLX module tree, and writes
``config.json`` + ``model.safetensors``.
"""

import json
import os
import re
import tarfile
import tempfile
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------
# Key-name transformations
# ------------------------------------------------------------------

def _rename_lstm_key(key: str) -> str | None:
    """Rename PyTorch LSTM weight/bias keys to MLX ``nn.LSTM`` convention.

    PyTorch uses:
        *.weight_ih_l{n}  ->  *.layers.{n}.Wx
        *.weight_hh_l{n}  ->  *.layers.{n}.Wh
        *.bias_ih_l{n}    ->  *.layers.{n}.bias   (summed with bias_hh)
        *.bias_hh_l{n}    ->  *.layers.{n}.bias   (summed with bias_ih)

    Returns the new key, or None if this is a bias_hh key (its value is
    accumulated into the bias_ih key's entry).
    """
    # weight_ih_l{n}
    m = re.search(r"weight_ih_l(\d+)$", key)
    if m:
        layer_idx = m.group(1)
        prefix = key[: m.start()]
        return f"{prefix}layers.{layer_idx}.Wx"

    # weight_hh_l{n}
    m = re.search(r"weight_hh_l(\d+)$", key)
    if m:
        layer_idx = m.group(1)
        prefix = key[: m.start()]
        return f"{prefix}layers.{layer_idx}.Wh"

    # bias_ih_l{n}
    m = re.search(r"bias_ih_l(\d+)$", key)
    if m:
        layer_idx = m.group(1)
        prefix = key[: m.start()]
        return f"{prefix}layers.{layer_idx}.bias"

    # bias_hh_l{n} -> same target as bias_ih (will be added)
    m = re.search(r"bias_hh_l(\d+)$", key)
    if m:
        layer_idx = m.group(1)
        prefix = key[: m.start()]
        return f"{prefix}layers.{layer_idx}.bias"

    return key


def _rename_nemo_key(key: str) -> str:
    """Rename top-level NeMo module prefixes to our module tree.

    NeMo checkpoint keys:
        encoder.*                -> encoder.*
        decoder.prediction.*     -> predict_net.*
        joint.joint_net.*        -> joint_net.*  (internal layers)
        joint.encoder.*          -> joint_net.encoder_proj.*
        joint.decoder.*          -> joint_net.decoder_proj.*
    """
    # decoder.prediction.embed -> predict_net.embed
    if key.startswith("decoder.prediction."):
        suffix = key[len("decoder.prediction."):]
        # decoder.prediction.embed.weight -> predict_net.embed.weight
        # decoder.prediction.dec_rnn.* -> predict_net.dec_rnn.*
        return f"predict_net.{suffix}"

    # joint.joint_net.{idx}.weight/bias -> joint_net.output.weight/bias
    # NeMo joint_net is a Sequential: [activation, identity, Linear(joint_hidden, vocab)]
    # The only trainable layer is the last Linear at index 2
    if key.startswith("joint.joint_net."):
        suffix = key[len("joint.joint_net."):]
        # e.g. "2.weight" -> "output.weight"
        m = re.match(r"(\d+)\.(.*)", suffix)
        if m:
            return f"joint_net.output.{m.group(2)}"
        return f"joint_net.output.{suffix}"

    # joint.encoder.weight -> joint_net.encoder_proj.weight
    if key.startswith("joint.encoder."):
        suffix = key[len("joint.encoder."):]
        return f"joint_net.encoder_proj.{suffix}"

    # joint.decoder.weight -> joint_net.decoder_proj.weight
    if key.startswith("joint.decoder."):
        suffix = key[len("joint.decoder."):]
        return f"joint_net.decoder_proj.{suffix}"

    # decoder.decoder_layers.* (CTC decoder if present) — keep as-is
    # encoder.* — keep as-is
    return key


# ------------------------------------------------------------------
# Tensor transformations
# ------------------------------------------------------------------

def _should_transpose_conv(key: str) -> bool:
    """True for Conv2d / Conv1d weight keys in the encoder or CTC decoder."""
    if "conv" in key or "ctc_decoder" in key:
        return True
    if key == "decoder.decoder_layers.0.weight":
        return True
    return False


def _transform_tensor(key: str, tensor: "np.ndarray") -> "np.ndarray":
    """Apply layout transformations for MLX compatibility."""
    if _should_transpose_conv(key):
        if tensor.ndim == 4:
            # PyTorch Conv2d: [out, in, H, W] -> MLX: [out, H, W, in]
            tensor = tensor.transpose(0, 2, 3, 1)
        elif tensor.ndim == 3:
            # PyTorch Conv1d: [out, in, K] -> MLX: [out, K, in]
            tensor = tensor.transpose(0, 2, 1)
    return tensor


# ------------------------------------------------------------------
# Main conversion
# ------------------------------------------------------------------

def convert_nemo_to_mlx(nemo_path: str, output_dir: str) -> None:
    """Convert a ``.nemo`` checkpoint to MLX safetensors format.

    Parameters
    ----------
    nemo_path : str
        Path to the ``.nemo`` file (a tar.gz archive).
    output_dir : str
        Directory to write ``config.json`` and ``model.safetensors`` into.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for weight conversion. "
            "Install it with: pip install torch"
        )

    try:
        from safetensors.numpy import save_file
    except ImportError:
        raise ImportError(
            "safetensors is required for weight conversion. "
            "Install it with: pip install safetensors"
        )

    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for weight conversion. "
            "Install it with: pip install pyyaml"
        )

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Extract .nemo tarball
        with tarfile.open(nemo_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        # 2. Locate checkpoint and config
        ckpt_path = _find_file(tmpdir, "model_weights.ckpt")
        config_path = _find_file(tmpdir, "model_config.yaml")

        if ckpt_path is None:
            raise FileNotFoundError(
                f"model_weights.ckpt not found in {nemo_path}"
            )
        if config_path is None:
            raise FileNotFoundError(
                f"model_config.yaml not found in {nemo_path}"
            )

        # 3. Load PyTorch weights
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # 4. Load config
        with open(config_path, "r") as f:
            nemo_config = yaml.safe_load(f)

        # 5. Transform weights
        new_state: dict[str, np.ndarray] = {}

        for key, value in state.items():
            # Skip preprocessor and batch-norm tracking
            if key.startswith("preprocessor."):
                continue
            if "_num_batches_tracked" in key:
                continue

            # Convert to numpy
            arr = value.numpy()

            # Tensor layout transforms (before key rename, keys still NeMo-style)
            arr = _transform_tensor(key, arr)

            # Key rename: NeMo module prefixes
            new_key = _rename_nemo_key(key)

            # Key rename: LSTM weight/bias names
            new_key = _rename_lstm_key(new_key)

            # For LSTM biases: bias_ih and bias_hh map to the same target
            # key and must be summed together.
            if new_key in new_state and new_key.endswith(".bias"):
                new_state[new_key] = new_state[new_key] + arr
            else:
                new_state[new_key] = arr

        # 6. Extract vocabulary
        vocab = _extract_vocab(nemo_config)

        # 7. Build config.json
        model_config = _build_config(nemo_config, vocab)

        # 8. Save
        config_out = os.path.join(output_dir, "config.json")
        with open(config_out, "w") as f:
            json.dump(model_config, f, indent=2)

        weights_out = os.path.join(output_dir, "model.safetensors")
        save_file(new_state, weights_out)

        print(f"Saved config to {config_out}")
        print(f"Saved weights to {weights_out}")
        print(f"  {len(new_state)} tensors")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_file(root: str, name: str) -> str | None:
    """Walk *root* looking for a file named *name*."""
    for dirpath, _, filenames in os.walk(root):
        if name in filenames:
            return os.path.join(dirpath, name)
    return None


def _extract_vocab(nemo_config: dict) -> list[str]:
    """Pull the BPE vocabulary list from the NeMo config."""
    # Try several known locations
    for path in [
        ("model", "decoder", "vocabulary"),
        ("model", "joint", "vocabulary"),
        ("decoder", "vocabulary"),
        ("joint", "vocabulary"),
    ]:
        node = nemo_config
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        if node is not None and isinstance(node, list):
            return node

    # Fallback: check for tokenizer model file
    print(
        "WARNING: Could not find vocabulary in config. "
        "You may need to add it manually to config.json."
    )
    return []


def _build_config(nemo_config: dict, vocab: list[str]) -> dict:
    """Build a minimal config.json for the MLX model."""
    # Navigate into "model" if it exists (NeMo configs often nest under "model")
    cfg = nemo_config.get("model", nemo_config)

    encoder_cfg = cfg.get("encoder", {})
    decoder_cfg = cfg.get("decoder", {})
    joint_cfg = cfg.get("joint", {})

    return {
        "model_type": "nemotron_asr",
        "encoder": {
            "d_model": encoder_cfg.get("d_model", 1024),
            "n_layers": encoder_cfg.get("n_layers", 24),
            "n_heads": encoder_cfg.get("n_heads", 8),
            "kernel_size": encoder_cfg.get("kernel_size", 9),
            "subsampling_factor": encoder_cfg.get("subsampling_factor", 8),
            "feat_in": encoder_cfg.get("feat_in", 128),
            "att_context_size": encoder_cfg.get("att_context_size", [70, 1]),
        },
        "decoder": {
            "vocab_size": decoder_cfg.get("vocab_size", 1024),
            "pred_hidden": decoder_cfg.get("prednet", {}).get("pred_hidden", 640),
            "pred_rnn_layers": decoder_cfg.get("prednet", {}).get(
                "pred_rnn_layers", 2
            ),
        },
        "joint": {
            "joint_hidden": joint_cfg.get("jointnet", {}).get("joint_hidden", 640),
            "encoder_hidden": joint_cfg.get("jointnet", {}).get(
                "encoder_hidden", 1024
            ),
        },
        "preprocessor": {
            "features": cfg.get("preprocessor", {}).get("features", 128),
            "sample_rate": cfg.get("preprocessor", {}).get("sample_rate", 16000),
            "window_size": cfg.get("preprocessor", {}).get("window_size", 0.025),
            "window_stride": cfg.get("preprocessor", {}).get("window_stride", 0.01),
            "normalize": cfg.get("preprocessor", {}).get("normalize", "NA"),
        },
        "vocabulary": vocab,
    }


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def convert_cli(nemo_path: str, output_dir: str) -> None:
    """CLI-callable wrapper around :func:`convert_nemo_to_mlx`."""
    convert_nemo_to_mlx(nemo_path, output_dir)
