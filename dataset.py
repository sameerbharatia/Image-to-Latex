import os
import re
from zipfile import ZipFile
from string import ascii_letters
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, ToTensor


# Constants for token indices
PAD, SOS, EOS, UNK = 0, 1, 2, 3
# Maximum RGB value, representing white
WHITE = 255

# Paths and directories
ZIP_DIR = 'data.zip'
DATA_DIR = 'data'
IMAGES_DIR = f'{DATA_DIR}/formula_images_processed'
TRAIN_DATA = f'{DATA_DIR}/im2latex_train.csv'
TEST_DATA = f'{DATA_DIR}/im2latex_test.csv'
VAL_DATA = f'{DATA_DIR}/im2latex_validate.csv'

# Batch size for data loading
BATCH_SIZE = 64
# Worker and memory settings for DataLoader
NUM_WORKERS = 0
PIN_MEMORY = False

# Environmental variables for distributed computing
RANK = int(os.getenv('SLURM_PROCID', 0))
WORLD_SIZE = int(os.getenv('SLURM_NTASKS', 1))

class Vocabulary:
    """A simple vocabulary mapping for tokenization and detokenization of text."""

    def __init__(self, train_csv: str) -> None:
        """
        Initialize the vocabulary using the training data.

        Args:
            train_csv (str): Path to the training CSV file.
        """
        # Initial mapping of special tokens
        self.token_to_index = {'<PAD>': PAD, '<SOS>': SOS, '<EOS>': EOS, '<UNK>': UNK}
        # Reverse mapping for decoding
        self.index_to_token = {index: token for token, index in self.token_to_index.items()}
        self.create_vocab(train_csv)
        self.size = len(self.token_to_index)
        print(f'Vocabulary of size {self.size} generated')

    def add_token(self, token: str) -> None:
        """Adds a token to the vocabulary if it is not already present."""
        if token not in self.token_to_index:
            index = len(self.token_to_index)
            self.token_to_index[token] = index
            self.index_to_token[index] = token

    def add_all_tokens(self, tokens: list[str]) -> None:
        """Adds multiple tokens to the vocabulary."""
        for token in tokens:
            self.add_token(token)

    def create_vocab(self, train_csv: str) -> None:
        """Generates the vocabulary based on the formulas present in the training data."""
        with ZipFile(f'{DATA_DIR}.zip', 'r') as data:
            formulas = pd.read_csv(data.open(train_csv))['formula']
        formulas = formulas.apply(self.clean_latex)
        formulas = formulas.str.split()
        formulas.apply(self.add_all_tokens)
        # Include all English letters in the vocabulary
        for letter in ascii_letters:
            self.add_token(letter)

    def encode(self, formula: str) -> list[int]:
        """Encodes a LaTeX formula into a list of token indices."""
        return [self.token_to_index['<SOS>']] + [self.token_to_index.get(token, UNK) for token in self.clean_latex(formula).split()] + [self.token_to_index['<EOS>']]

    def decode(self, indices: list[int]) -> list[str]:
        """Decodes a list of token indices back into a list of tokens."""
        result = []
        eos = self.token_to_index['<EOS>']
        for index in indices:
            result.append(self.index_to_token.get(int(index), '<UNK>'))
            if index == eos:
                break
        return result

    @staticmethod
    def clean_latex(latex_str: str) -> str:
        """Cleans a LaTeX string by simplifying certain commands and removing spacing."""
        # Simplify LaTeX commands for left and right delimiters
        latex_str = re.sub(r'\\left\(|\\right\)', '()', latex_str)
        latex_str = re.sub(r'\\left\[|\\right\]', '[]', latex_str)
        latex_str = re.sub(r'\\left\{|\\right\}', '{}', latex_str)
        # Remove various LaTeX spacing commands
        for cmd in [r'\\,', r'\\:', r'\\;', r'\\!', r'\\ ', r'\\thinspace', r'\\medspace', r'\\thickspace', r'\\enspace', r'\\hspace', r'\\vspace']:
            latex_str = re.sub(fr'{cmd}(\{{.*?\}})?', '', latex_str)
        return latex_str

class LatexDataset(Dataset):
    """A custom dataset class for loading LaTeX formula images and their corresponding tokenized representations."""

    def __init__(self, zip_file: str, csv_file: str, image_dir: str, vocab: Vocabulary, transform: Optional[Callable] = None) -> None:
        """
        Initializes the dataset.

        Args:
            zip_file (str): Path to the zip file containing images and the CSV file.
            csv_file (str): Name of the CSV file inside the zip file.
            image_dir (str): Directory name inside the zip file containing images.
            vocab (Vocabulary): An instance of the Vocabulary class for encoding formulas.
            transform (Optional[Callable], optional): A function/transform that takes in an PIL image and returns a transformed version.
        """
        self.zip_file = ZipFile(zip_file, 'r')
        self.annotations = pd.read_csv(self.zip_file.open(csv_file))
        self.image_dir = image_dir
        self.transform = transform
        self.vocab = vocab

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, list[int]]:
        """Retrieves an image and its corresponding tokenized formula by index."""
        img_name = f'{self.image_dir}/{self.annotations.loc[idx, "image"]}'
        image = Image.open(self.zip_file.open(img_name)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        latex_formula = self.annotations.loc[idx, 'formula']
        indices = self.vocab.encode(latex_formula)
        return image, indices

    def close(self) -> None:
        """Closes the zip file."""
        self.zip_file.close()

def collate_fn(batch: list[Tuple[Image.Image, list[int]]]) -> Tuple[Tensor, Tensor]:
    """
    A function to collate data into batches for the DataLoader.

    Args:
        batch (list[Tuple[Image.Image, list[int]]]): A batch of tuples of images and sequences.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing a batch of images and a batch of padded sequences.
    """
    images = [image for image, _ in batch]
    sequences = [torch.tensor(indices) for _, indices in batch]
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=PAD)
    images = torch.stack(images, dim=0)
    return images, sequences_padded

class PadToMaxSize:
    """A transform to pad images to the maximum size found in the dataset."""

    def __init__(self, train_images_dir: str) -> None:
        """Finds the maximum image dimensions in the dataset."""
        self.max_height, self.max_width = self._find_max_dimensions(train_images_dir)
        print(f'Max height by width is {self.max_height} by {self.max_width}')

    def __call__(self, img: Image.Image) -> Image.Image:
        """Applies padding to the image to match the maximum dimensions."""
        return self._pad_to_max(img, self.max_height, self.max_width)

    def _find_max_dimensions(self, train_images_dir: str) -> Tuple[int, int]:
        """Finds the maximum width and height among all images in the dataset."""
        max_width, max_height = 0, 0
        with ZipFile(ZIP_DIR, 'r') as data:
            for file in data.namelist():
                if file.startswith(train_images_dir) and file.endswith('.png'):
                    with Image.open(data.open(file)) as img:
                        width, height = img.size
                        max_width, max_height = max(max_width, width), max(max_height, height)
        return max_height, max_width

    def _pad_to_max(self, img: Image.Image, max_height: int, max_width: int, fill: int = WHITE, padding_mode: str = 'constant') -> Image.Image:
        """Pads an image to the maximum dimensions, centered."""
        aspect_ratio = img.width / img.height
        if img.width > img.height:
            new_width, new_height = max_width, int(max_width / aspect_ratio)
        else:
            new_height, new_width = max_height, int(max_height * aspect_ratio)
        resized_img = F.resize(img, (new_height, new_width))
        padding_left, padding_top = (max_width - new_width) // 2, (max_height - new_height) // 2
        padding_right, padding_bottom = max_width - new_width - padding_left, max_height - new_height - padding_top
        return F.pad(resized_img, (padding_left, padding_top, padding_right, padding_bottom), fill=fill, padding_mode=padding_mode)

# Initialization of vocabulary and datasets with transformations
vocab = Vocabulary(TRAIN_DATA)
transform = Compose([
    PadToMaxSize(IMAGES_DIR),
    ToTensor(),
])

train_dataset = LatexDataset(zip_file=ZIP_DIR, csv_file=TRAIN_DATA, image_dir=IMAGES_DIR, vocab=vocab, transform=transform)
test_dataset = LatexDataset(zip_file=ZIP_DIR, csv_file=TEST_DATA, image_dir=IMAGES_DIR, vocab=vocab, transform=transform)
val_dataset = LatexDataset(zip_file=ZIP_DIR, csv_file=VAL_DATA, image_dir=IMAGES_DIR, vocab=vocab, transform=transform)

def get_dataloader(rank: int, world_size: int, dataset: Dataset) -> DataLoader:
    """
    Prepares a DataLoader for distributed training.

    Args:
        rank (int): The rank of the current process in the distributed setup.
        world_size (int): The total number of processes in the distributed setup.
        dataset (Dataset): The dataset to load data from.

    Returns:
        DataLoader: A DataLoader instance ready for distributed training.
    """
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn, sampler=sampler)