import os
import random
import datetime
import numpy as np
import torch
import torch.distributed as dist


# ==================== Device Detection ====================

def get_device():
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_distributed():
    """Check if running in distributed mode (launched via torchrun)."""
    return "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_training(seed=42):
    """Setup training environment. Returns (device, rank, world_size, is_ddp).

    If launched with torchrun: initializes DDP, returns CUDA device.
    Otherwise: returns best available single device (MPS/CPU).
    """
    if is_distributed():
        rank, world_size, local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        set_seed(seed, rank)
        return device, rank, world_size, True
    else:
        device = get_device()
        set_seed(seed, 0)
        print(f"[Single Device] Using {device}")
        return device, 0, 1, False


# ==================== DDP (kept for multi-GPU compatibility) ====================

def setup_ddp():
    """Initializes the distributed data parallel environment."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(
        f"[DDP Setup] Global Rank: {rank}/{world_size}, "
        f"Local Rank (GPU): {local_rank} on device {torch.cuda.current_device()}"
    )
    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ==================== Utilities ====================

def set_seed(seed: int, rank: int = 0):
    """Sets the random seed for reproducibility."""
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    """Returns human-readable trainable parameter count."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"
    else:
        return f"{total_params / 1e3:.1f}K"


def reduce_tensor(tensor: torch.Tensor, world_size: int, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """Reduces a tensor across all processes (DDP only)."""
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    if op == dist.ReduceOp.AVG:
        rt /= world_size
    return rt


def format_time(seconds: float) -> str:
    """Formats seconds into H:M:S string."""
    return str(datetime.timedelta(seconds=int(seconds)))
