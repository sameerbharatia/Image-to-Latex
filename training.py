import os
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from dataset import vocab


def setup(rank: int, world_size: int) -> None:
    """
    Initialize the distributed environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes in the distributed setup.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup() -> None:
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def warmup_cosine_annealing_lr(init_lr: float, num_warmup_steps: int, num_total_steps: int, num_cycles: float = 0.5) -> Callable[[int], float]:
    """
    Generate a learning rate scheduler based on warmup and cosine annealing.

    Args:
        init_lr (float): Initial learning rate.
        num_warmup_steps (int): Number of steps for the linear warm-up.
        num_total_steps (int): Total number of training steps.
        num_cycles (float): Fraction of the total steps where the cosine annealing is active.

    Returns:
        Callable[[int], float]: A function to compute the learning rate based on the current step.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps)) * init_lr
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_total_steps - num_warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.pi * num_cycles * 2.0 * progress)) * init_lr

    return lr_lambda

def train_for_epoch(rank: int, world_size: int, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: LambdaLR, scaler: GradScaler, train_data: DataLoader) -> Optional[float]:
    """
    Train the model for one epoch in a distributed setup.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes in the distributed setup.
        model (nn.Module): The neural network model to train.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimization algorithm.
        scheduler (LambdaLR): Learning rate scheduler.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        train_data (DataLoader): DataLoader for the training data.

    Returns:
        Optional[float]: Average loss for this training epoch, computed across all processes.
    """
    model.train()
    local_loss = 0.0

    for images, sequences in train_data:
        images, sequences = images.to(rank), sequences.to(rank)
        optimizer.zero_grad()

        with autocast():
            outputs = model(images, sequences)
            loss = criterion(outputs.reshape(-1, vocab.size), sequences.flatten())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        local_loss += loss.item() * images.size(0)

    # Aggregate loss across all processes
    total_loss_tensor = torch.tensor(local_loss).to(rank)
    dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_loss = total_loss_tensor.item() / (len(train_data.dataset) * world_size)
        return avg_loss
    return None

def gather_tensors_on_rank0(tensor: torch.Tensor, world_size: int, rank: int) -> Optional[list[torch.Tensor]]:
    """
    Gathers tensors from all ranks to rank 0.

    Args:
        tensor (torch.Tensor): The tensor to gather.
        world_size (int): The total number of processes in the distributed setup.
        rank (int): The rank of the current process.

    Returns:
        Optional[list[torch.Tensor]]: A list of tensors gathered on rank 0, None on other ranks.
    """
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)] if rank == 0 else None
    dist.gather(tensor, gather_list=gathered_tensors, dst=0)
    return gathered_tensors

def evaluate_model(rank: int, world_size: int, model: nn.Module, data: DataLoader, criterion: nn.Module, vocab) -> tuple[Optional[float], Optional[float]]:
    """
    Evaluate the model performance on the validation set.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes in the distributed setup.
        model (nn.Module): The neural network model.
        data (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): The loss function.
        vocab: Vocabulary mapping for decoding.

    Returns:
        tuple[Optional[float], Optional[float]]: Average loss and BLEU score, computed on rank 0. None for other ranks.
    """
    model.eval()
    local_loss = 0.0

    predictions = torch.tensor([], dtype=torch.long, device=rank)
    references = []

    with torch.no_grad():
        for images, sequences in data:
            images, sequences = images.to(rank), sequences.to(rank)
            outputs = model(images, sequences)
            loss = criterion(outputs.reshape(-1, vocab.size), sequences.flatten())
            local_loss += loss.item() * images.size(0)

            batch_predictions = model.module.predict(images) if hasattr(model, 'module') else model.predict(images)
            predictions = torch.cat((predictions, batch_predictions), dim=0)
            references.extend(sequences.tolist())

    total_loss_tensor = torch.tensor(local_loss, device=rank)
    dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_loss = total_loss_tensor.item() / (len(data.dataset) * world_size)
        bleu_score = compute_bleu_score(predictions, references, vocab)
        return avg_loss, bleu_score
    return None, None

def compute_bleu_score(predictions: torch.Tensor, references: list[list[int]], vocab) -> float:
    """
    Computes the BLEU score for the given predictions and references.

    Args:
        predictions (torch.Tensor): Model predictions.
        references (list[list[int]]): Ground truth sequences.
        vocab: Vocabulary mapping for decoding.

    Returns:
        float: The computed BLEU score.
    """
    decoded_predictions = [vocab.decode(pred) for pred in predictions]
    decoded_references = [[vocab.decode(ref)] for ref in references]
    bleu_score = corpus_bleu(decoded_references, decoded_predictions, smoothing_function=SmoothingFunction().method7)
    return bleu_score

def save_model(model: nn.Module, file_path: str) -> None:
    """
    Save the model state to a file, compatible with both DDP and non-DDP contexts.

    Args:
        model (nn.Module): The model (or DDP-wrapped model) to save.
        file_path (str): Path to the file where the model state is saved.
    """
    model_to_save = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    torch.save(model_to_save.state_dict(), f'{file_path}.pt')

def load_model(model: nn.Module, file_path: str) -> None:
    """
    Load the model state from a file, compatible with both DDP and non-DDP contexts.

    Args:
        model (nn.Module): The model (or DDP-wrapped model) to load the state into.
        file_path (str): Path to the file from where to load the model state.
    """
    state_dict = torch.load(f'{file_path}.pt', map_location='cpu')
    # Remove 'module.' prefix added by nn.DataParallel
    new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())
    model_to_load = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    model_to_load.load_state_dict(new_state_dict)

def train_model(rank: int, world_size: int, model: nn.Module, train_data: DataLoader, val_data: DataLoader, criterion: nn.Module, max_epochs: int, learning_rate: float, file_path: str) -> tuple[Optional[nn.Module], list[float], list[float], list[float]]:
    """
    Trains and validates the model across multiple epochs in a distributed computing environment and keeps track of the best model based on validation performance.

    This function orchestrates the training process, involving setting up optimizers, schedulers,
    performing training and validation each epoch, and saving the model that achieves the best validation performance.

    Args:
        rank (int): The process rank within the distributed setup.
        world_size (int): The total number of processes participating in the computation.
        model (nn.Module): The model to be trained and validated.
        train_data (DataLoader): DataLoader providing the training data.
        val_data (DataLoader): DataLoader providing the validation data.
        criterion (nn.Module): Loss function to optimize.
        max_epochs (int): Maximum number of epochs to train.
        learning_rate (float): Initial learning rate for the optimizer.
        file_path (str): Path to save the model that achieves the best validation performance.

    Returns:
        tuple[Optional[nn.Module], list[float], list[float], list[float]]: A tuple containing the best model based on validation performance, 
        lists containing training losses, validation losses, and validation BLEU scores for each epoch, respectively. 
        For ranks other than 0, the best model is None, and lists are empty.
    """
    # Initialize training history lists and the best model placeholder
    train_losses, val_losses, val_bleus = [], [], []
    best_model = None
    min_val_loss = torch.inf  # Initialize the best validation loss

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Calculate the total steps for the learning rate scheduler
    num_batches = len(train_data)
    num_total_steps = max_epochs * num_batches
    num_warmup_steps = num_batches  # Linear warm-up during the first epoch

    # Initialize the learning rate scheduler with warmup and cosine annealing
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_annealing_lr(
        learning_rate, num_warmup_steps, num_total_steps
    ))

    # Set up the gradient scaler for mixed precision training
    scaler = GradScaler()

    # Training loop across epochs
    for epoch in range(1, max_epochs + 1):
        # Adjust data samplers for distributed training
        train_data.sampler.set_epoch(epoch)
        val_data.sampler.set_epoch(epoch)

        # Conduct training for one epoch
        train_loss = train_for_epoch(rank, world_size, model, criterion, optimizer, scheduler, scaler, train_data)
        if rank == 0:
            train_losses.append(train_loss)

        # Conduct validation after the training epoch
        val_loss, val_bleu = evaluate_model(rank, world_size, model, val_data, criterion, vocab, epoch)
        if rank == 0:
            val_losses.append(val_loss)
            val_bleus.append(val_bleu)

            # Update the best model if current validation performance is improved
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = model
                save_model(model, file_path)  # Save the currently best model

            # Log the training and validation progress
            print(f'Epoch {epoch:<3} Train Loss: {train_loss:<6.4f} (or "N/A"), Validation Loss: {val_loss:<6.4f}, Validation BLEU: {val_bleu:<6.4f}')

    # Return the best model based on validation performance along with the training history
    # For non-zero ranks, best_model is None and lists are empty
    return (best_model, train_losses, val_losses, val_bleus) if rank == 0 else (None, [], [], [])