import torch
import torch.nn as nn
import pandas as pd

from dataset import vocab, PAD
from dataset import train_dataset, val_dataset, test_dataset, get_dataloader

from model import Im2Latex

from training import train_model, evaluate_model
from training import setup, cleanup

from torch.nn.parallel import DistributedDataParallel as DDP
import os


# Global training settings
EPOCHS = 20
LEARNING_RATE = 0.001

def main(rank: int, world_size: int) -> None:
    """
    Main training function for distributed training.

    Args:
        rank (int): The current process ID within the world.
        world_size (int): The total number of processes participating in the training.
    """
    # Setup distributed environment
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Initialize the model and wrap it for distributed training
    model = Im2Latex(
        num_decoder_layers=3,
        hidden_dim=128,
        ff_dim=256,
        num_heads=4,
        dropout=0.3,
        max_out_length=150,
        vocab_size=vocab.size
    ).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Log model info on the main process
    if rank == 0:
        num_gpus = torch.cuda.device_count()
        print(f'Working with {num_gpus} GPUs')
        num_parameters = sum(p.numel() for p in model.parameters())
        print(f'Training Image to LaTeX model with {num_parameters} parameters')

    # Prepare data loaders for training and validation datasets
    train_loader = get_dataloader(rank, world_size, train_dataset)
    val_loader = get_dataloader(rank, world_size, val_dataset)
    test_loader = get_dataloader(rank, world_size, test_dataset)

    # Define the loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # Train the model
    try:
        best_model, train_losses, val_losses, val_bleus = train_model(
            rank, world_size, model, train_loader, val_loader, criterion, EPOCHS, LEARNING_RATE, 'saved_models/image_to_latex_model'
        )
    finally:
        # Cleanup and close datasets
        cleanup()
        train_loader.dataset.close()
        val_loader.dataset.close()

    # Save training metrics to a CSV file on the main process
    if rank == 0:
        training_metrics = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_bleu': val_bleus
        }
        pd.DataFrame(training_metrics).to_csv('training_metrics.csv')

    test_loss, test_bleu = evaluate_model(rank, world_size, best_model, test_loader, criterion, vocab)

    if rank == 0:
        print(f'Test Loss: {test_loss if test_loss is not None else "N/A":<6.4f}, Test BLEU: {test_bleu:<6.4f}')

if __name__ == '__main__':
    rank = int(os.getenv('SLURM_PROCID', '0'))
    world_size = int(os.getenv('SLURM_NTASKS', '1'))
    # Configuration to optimize CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    main(rank, world_size)