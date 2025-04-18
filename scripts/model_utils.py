import os
import re
import glob

def find_best_val_checkpoint(log_dir="lightning_logs"):
    """
    Find the checkpoint with the lowest validation loss.
    """
    pattern = os.path.join(log_dir, "version_*", "checkpoints", "*.ckpt")
    checkpoint_files = glob.glob(pattern)

    best_ckpt = None
    lowest_val_loss = float("inf")
    val_loss_pattern = re.compile(r"val_loss=([\d\.eE+-]+)\.ckpt")

    for ckpt_path in checkpoint_files:
        match = val_loss_pattern.search(ckpt_path)
        if match:
            val_loss = float(match.group(1))
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_ckpt = ckpt_path

    if best_ckpt:
        return best_ckpt
    else:
        raise FileNotFoundError("No checkpoint files with val_loss found.")