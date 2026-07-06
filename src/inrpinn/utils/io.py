"""Checkpoint and config I/O.

  save_checkpoint(model, optimiser, epoch, loss, path)
  load_checkpoint(path, model, optimiser) -> epoch, loss
    Standard PyTorch state-dict checkpointing.

  load_config(yaml_path) -> dict
    Loads a YAML config file and validates required keys.
    Merges base defaults before returning the final config dict.

  save_metrics(metrics_dict, path)
    Appends a row to a CSV log file (one row per epoch).
"""
