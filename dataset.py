import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, ToTensor

def collate_fn(batch):
    images = [item['image'] for item in batch]
    sequences = [torch.tensor(item['indices']) for item in batch]
    
    # Pad sequences to have the same length
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Stack images into a single tensor
    images = torch.stack(images, dim=0)
    
    return {'image': images, 'indices': sequences_padded}

class Vocabulary:
    def __init__(self, token_to_index=None):
        if token_to_index is None:
            self.token_to_index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        else:
            self.token_to_index = token_to_index
        self.index_to_token = {index: token for token, index in self.token_to_index.items()}

    def add_token(self, token):
        if token not in self.token_to_index:
            index = len(self.token_to_index)
            self.token_to_index[token] = index
            self.index_to_token[index] = token

    def tokenize(self, formula):
        return [self.token_to_index.get(token, self.token_to_index["<UNK>"]) for token in formula.split()]

class Latex_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, vocab, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            vocab (Vocabulary): An instance of the Vocabulary class.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')
        latex_formula = self.annotations.iloc[idx, 0]
        indices = self.vocab.tokenize(latex_formula)
        # pad up to length 512
        sample = {'image': image, 'indices': indices}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
    
def pad_to_max(img, max_width, max_height, fill=255, padding_mode='constant'):
    # First, resize the image to maintain aspect ratio
    aspect_ratio = img.width / img.height
    if img.width > img.height:
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(max_height * aspect_ratio)

    resized_img = F.resize(img, (new_height, new_width))

    # Calculate padding
    padding_left = (max_width - new_width) // 2
    padding_top = (max_height - new_height) // 2
    padding_right = max_width - new_width - padding_left
    padding_bottom = max_height - new_height - padding_top

    # Pad the resized image
    padded_img = F.pad(resized_img, (padding_left, padding_top, padding_right, padding_bottom), fill=fill, padding_mode=padding_mode)
    return padded_img

class PadToMaxSize(object):
    def __init__(self, max_width, max_height):
        self.max_width = max_width
        self.max_height = max_height

    def __call__(self, img):
        return pad_to_max(img, self.max_width, self.max_height)