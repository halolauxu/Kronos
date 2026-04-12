import os
import sys
import glob
import json
import time
from time import gmtime, strftime
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import comet_ml
except ImportError:
    comet_ml = None

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import (
    setup_training,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time,
)


def create_dataloaders(config: dict, rank: int, world_size: int, is_ddp: bool):
    """
    Creates and returns dataloaders for training and validation.

    In DDP mode, uses DistributedSampler. In single-device mode, uses regular
    DataLoader with shuffle.

    Args:
        config (dict): A dictionary of configuration parameters.
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.
        is_ddp (bool): Whether running in distributed data parallel mode.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, train_dataset, valid_dataset).
    """
    mode_str = "distributed" if is_ddp else "single-device"
    print(f"[Rank {rank}] Creating {mode_str} dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            shuffle=False,  # Shuffle is handled by the sampler
            num_workers=config.get('num_workers', 2),
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=config['batch_size'],
            sampler=val_sampler,
            shuffle=False,
            num_workers=config.get('num_workers', 2),
            pin_memory=True,
            drop_last=False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 2),
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 2),
            pin_memory=True,
            drop_last=False
        )

    print(f"[Rank {rank}] Dataloaders created. Train steps/epoch: {len(train_loader)}, Val steps: {len(val_loader)}")
    return train_loader, val_loader, train_dataset, valid_dataset


def _get_raw_model(model, is_ddp):
    """Return the underlying model, unwrapping DDP if needed."""
    return model.module if is_ddp else model


def _find_latest_checkpoint(save_dir):
    """Find the latest epoch checkpoint in save_dir/checkpoints/epoch_N/.

    Returns (epoch_number, checkpoint_path) or (None, None) if none found.
    """
    ckpt_root = os.path.join(save_dir, "checkpoints")
    pattern = os.path.join(ckpt_root, "epoch_*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None, None
    # Extract epoch numbers and pick the largest
    epochs = []
    for m in matches:
        basename = os.path.basename(m)
        try:
            epoch_num = int(basename.split("_")[1])
            epochs.append((epoch_num, m))
        except (IndexError, ValueError):
            continue
    if not epochs:
        return None, None
    epochs.sort(key=lambda x: x[0])
    return epochs[-1]


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size, is_ddp):
    """
    The main training and validation loop for the predictor.

    Args:
        model: The model to train (DDP-wrapped in distributed mode, plain otherwise).
        tokenizer: The frozen tokenizer for encoding inputs.
        device (torch.device): The device for the current process.
        config (dict): Configuration dictionary.
        save_dir (str): Directory to save checkpoints.
        logger: Comet logger instance (or None).
        rank (int): Global rank of the process.
        world_size (int): Total number of processes.
        is_ddp (bool): Whether running in distributed data parallel mode.

    Returns:
        dict: A dictionary of training results.
    """
    start_time = time.time()
    is_main = (rank == 0)

    if is_main:
        effective_bs = config['batch_size'] * world_size
        device_label = "per GPU" if is_ddp else "per device"
        print(f"[Rank {rank}] BATCHSIZE ({device_label}): {config['batch_size']}")
        print(f"[Rank {rank}] Effective total batch size: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(
        config, rank, world_size, is_ddp
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['predictor_learning_rate'],
        steps_per_epoch=len(train_loader), epochs=config['epochs'],
        pct_start=0.03, div_factor=10
    )

    # --- Checkpoint resume ---
    start_epoch = 0
    best_val_loss = float('inf')
    latest_epoch, latest_ckpt_path = _find_latest_checkpoint(save_dir)
    if latest_ckpt_path is not None:
        ckpt_file = os.path.join(latest_ckpt_path, "training_state.pt")
        if os.path.isfile(ckpt_file):
            print(f"Resuming from epoch {latest_epoch + 1}")
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            raw_model = _get_raw_model(model, is_ddp)
            raw_model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            print(f"Checkpoint loaded. Starting at epoch {start_epoch + 1}/{config['epochs']}")

    dt_result = {}
    batch_idx_global = start_epoch * len(train_loader)

    for epoch_idx in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        model.train()

        if is_ddp:
            train_loader.sampler.set_epoch(epoch_idx)

        # Set dataset seeds for reproducible sampling
        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)  # Keep validation sampling consistent

        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.squeeze(0).to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.squeeze(0).to(device, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation
            raw_model = _get_raw_model(model, is_ddp)
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = raw_model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            # Logging (Master Process Only)
            if is_main and (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                )
            if is_main and logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_batches_processed_rank = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.squeeze(0).to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.squeeze(0).to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                raw_model = _get_raw_model(model, is_ddp)
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                val_loss, _, _ = raw_model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                tot_val_loss_sum_rank += val_loss.item()
                val_batches_processed_rank += 1

        # Aggregate validation losses
        if is_ddp:
            val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
            val_batches_tensor = torch.tensor(val_batches_processed_rank, device=device)
            dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_sum_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0
        else:
            avg_val_loss = tot_val_loss_sum_rank / val_batches_processed_rank if val_batches_processed_rank > 0 else 0

        # --- End of Epoch Summary & Checkpointing ---
        if is_main:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)

            raw_model = _get_raw_model(model, is_ddp)

            # Save epoch checkpoint (for resume)
            epoch_ckpt_path = os.path.join(save_dir, "checkpoints", f"epoch_{epoch_idx}")
            os.makedirs(epoch_ckpt_path, exist_ok=True)
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(epoch_ckpt_path, "training_state.pt"))
            print(f"Epoch checkpoint saved to {epoch_ckpt_path}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                raw_model.save_pretrained(save_path)
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
                if logger:
                    logger.log_model("best_model", save_path)

        if is_ddp:
            dist.barrier()  # Ensure all processes finish the epoch before starting the next one.

    dt_result['best_val_loss'] = best_val_loss
    return dt_result


def main(config: dict):
    """
    Main function to orchestrate training.

    Supports both distributed (torchrun) and single-device (python3) execution.
    """
    device, rank, world_size, is_ddp = setup_training(config['seed'])
    is_main = (rank == 0)

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

    # Logger and summary setup (master process only)
    comet_logger, master_summary = None, {}
    if is_main:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
            'device': str(device),
        }
        if config.get('use_comet', False) and comet_ml is not None:
            comet_logger = comet_ml.Experiment(
                api_key=config['comet_config']['api_key'],
                project_name=config['comet_config']['project_name'],
                workspace=config['comet_config']['workspace'],
            )
            comet_logger.add_tag(config['comet_tag'])
            comet_logger.set_name(config['comet_name'])
            comet_logger.log_parameters(config)
            print("Comet Logger Initialized.")

    if is_ddp:
        dist.barrier()  # Ensure save directory is created before proceeding

    # Load frozen tokenizer
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device)

    # Model Initialization
    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    model.to(device)

    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main:
        raw_model = _get_raw_model(model, is_ddp)
        print(f"Predictor Model Size: {get_model_size(raw_model)}")

    # Start Training
    dt_result = train_model(
        model, tokenizer, device, config, save_dir, comet_logger, rank, world_size, is_ddp
    )

    # Finalize and save summary (master process only)
    if is_main:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger:
            comet_logger.end()

    if is_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    # Usage:
    #   DDP:           torchrun --standalone --nproc_per_node=NUM_GPUS train_predictor.py
    #   Single device: python3 train_predictor.py
    config_instance = Config()
    main(config_instance.__dict__)
