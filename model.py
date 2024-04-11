import math

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torchvision.models as models

from dataset import PAD, SOS, EOS


class PositionalEncoding1D(nn.Module):
    """
    Implements positional encoding for a 1D input.
    """
    def __init__(self, hidden_dim: int, dropout: float, max_len: int) -> None:
        """
        Args:
            hidden_dim: Dimension of the hidden layer.
            dropout: Dropout rate.
            max_len: Maximum length of the input sequences.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = self.create_positional_encoding(max_len, hidden_dim)
        self.register_buffer('pe', pe)

    @staticmethod
    def create_positional_encoding(max_len: int, hidden_dim: int) -> torch.Tensor:
        """
        Creates the positional encoding.

        Args:
            max_len: Maximum length of the input sequences.
            hidden_dim: Dimension of the hidden layer.

        Returns:
            Positional encoding as a tensor.
        """
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input.

        Args:
            x: Input tensor of shape `(Batch Size, Sequence Length, Hidden Dimension)`.

        Returns:
            Tensor after applying positional encoding and dropout.
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class PositionalEncoding2D(nn.Module):
    """
    Implements positional encoding for a 2D input.
    """
    def __init__(self, hidden_dim: int, height: int, width: int) -> None:
        """
        Args:
            hidden_dim: Dimension of the hidden layer.
            height: Height of the input images.
            width: Width of the input images.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        assert hidden_dim % 2 == 0, 'Hidden dimension must be divisible by 2.'
        pe = self.create_positional_encoding(hidden_dim, height, width)
        self.register_buffer('pe', pe)

    @staticmethod
    def create_positional_encoding(hidden_dim: int, height: int, width: int) -> torch.Tensor:
        """
        Creates the positional encoding for 2D inputs.

        Args:
            hidden_dim: Dimension of the hidden layer.
            height: Height of the input images.
            width: Width of the input images.

        Returns:
            Positional encoding as a tensor.
        """
        # Split the hidden dimension into two for height and width respectively
        pe_height = PositionalEncoding1D.create_positional_encoding(height, hidden_dim // 2).squeeze(0).permute(1, 0).unsqueeze(2).expand(-1, -1, width)
        pe_width = PositionalEncoding1D.create_positional_encoding(width, hidden_dim // 2).squeeze(0).permute(1, 0).unsqueeze(1).expand(-1, height, -1)
        pe = torch.cat([pe_height, pe_width], dim=-1).flatten(start_dim=1)
        pe = pe.permute(1, 2, 0).unsqueeze(0)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input.

        Args:
            x: Input tensor of shape `(Batch Size, Channel, Height, Width)`.

        Returns:
            Tensor after applying positional encoding.
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

class Im2Latex(nn.Module):
    """
    Model to convert images to LaTeX sequences using a ResNet encoder and a Transformer decoder.
    """
    def __init__(self, num_decoder_layers: int, hidden_dim: int, ff_dim: int, num_heads: int, max_out_length: int, vocab_size: int, dropout: float = 0.1) -> None:
        """
        Initializes the Im2Latex model components.

        Args:
            num_decoder_layers: Number of decoder layers in the transformer.
            hidden_dim: Dimension of the hidden layer.
            ff_dim: Dimension of the feed forward network model.
            num_heads: Number of heads in the multi-head attention models.
            max_out_length: Maximum output sequence length.
            vocab_size: Size of the vocabulary.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_out_length = max_out_length
        assert max_out_length % 2 == 0, 'Maximum output length must be divisible by 2.'

        # Encoder: ResNet18 without the final few layers
        resnet18 = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet18.children())[:-3])
        self.project = nn.Conv2d(256, hidden_dim, 1)
        self.pos_encoder_2d = PositionalEncoding2D(hidden_dim, 50, 50)  # Fixed dimensions for simplicity

        # Decoder: Transformer with positional encoding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder_1d = PositionalEncoding1D(hidden_dim, dropout, 2000)  # Arbitrary max sequence length
        self.causal_mask = nn.Transformer.generate_square_subsequent_mask(2000)
        decoder_layer = TransformerDecoderLayer(hidden_dim, num_heads, ff_dim, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        """
        Encodes input images using the ResNet encoder and positional encoding.

        Args:
            x: Input tensor of images with shape `(Batch Size, Channel, Height, Width)`.

        Returns:
            Encoded output tensor.
        """
        encoder_out = self.encoder(x)                                       # (B, 256, 50, 50)
        encoder_out = self.project(encoder_out)                             # (B, H, 50, 50)
        encoder_out = self.pos_encoder_2d(encoder_out)                      # (B, H, 50, 50)
        encoder_out = encoder_out.flatten(start_dim=2)                      # (B, H, 50 * 50)
        # put sequence length first
        encoder_out = encoder_out.permute(2, 0, 1)                          # (50 * 50, B, H)

        return encoder_out
    
    def decode(self, encoder_out, tgt):
        """
        Decodes the encoded images to LaTeX sequences.

        Args:
            encoder_out: Output from the encoder.
            tgt: Target sequences for training.

        Returns:
            Decoded output tensor.
        """
        tgt_embeddings = self.embedding(tgt) * math.sqrt(self.hidden_dim)   # (B, S)
        tgt_embeddings = self.pos_encoder_1d(tgt_embeddings)                # (B, S)
        # put sequence length first
        tgt_embeddings = tgt_embeddings.transpose(0, 1)                     # (S, B)
        seq_len = tgt.shape[1]
        causal_mask = self.causal_mask[:seq_len, :seq_len]                  # (S, S)
        
        # (S, B, H)
        decoder_out = self.decoder(tgt_embeddings, encoder_out, tgt_mask=causal_mask, tgt_is_causal=True)
        output = self.output_layer(decoder_out)                             # (S, B, V)
        output = output.transpose(0, 1)                                     # (B, S, V)

        return output

    def forward(self, x, tgt):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of images with shape `(Batch Size, Channel, Height, Width)`.
            tgt: Target sequences for training.

        Returns:
            Output tensor of the model.
        """
        encoder_out = self.encode(x)
        output = self.decode(encoder_out, tgt)

        return output

    def predict(self, images):
        """
        Predict LaTeX sequences from input images.

        Args:
            images: Input tensor of images with shape `(Batch Size, Channel, Height, Width)`.

        Returns:
            Predicted sequences of tokens.
        """
        encoder_out = self.encode(images)
        predicted_tokens = self.generate_sequence(encoder_out)
        return predicted_tokens
    
    def generate_sequence(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        Generates LaTeX sequences given encoder output, one token at a time.

        Args:
            encoder_out: Encoded image features with shape `(Sequence Length, Batch Size, Hidden Dimension)`.

        Returns:
            Generated sequences of tokens with shape `(Batch Size, Max Output Length)`.
        """
        batch_size = encoder_out.size(1)
        device = encoder_out.device

        # Initialize the target sequence with SOS tokens
        tgt = torch.full((batch_size, 1), SOS, dtype=torch.long, device=device)

        # Container for the generated sequence
        generated_seq = torch.zeros((batch_size, 0), dtype=torch.long, device=device)

        # Mask to keep track of sequences that have already produced an EOS token
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.max_out_length):
            # Decode the current target sequence to get logits for the next token
            output = self.decode(encoder_out, tgt)

            # Select the last token's logits and convert them to probabilities
            last_token_logits = output[:, -1, :]
            last_token_ids = last_token_logits.argmax(dim=-1, keepdim=True)

            # Update sequences that have produced an EOS token
            just_finished = (last_token_ids.squeeze(-1) == EOS)
            active_sequences &= ~just_finished

            # For sequences that just finished, replace the token with PAD
            last_token_ids[just_finished, :] = PAD

            # Append the generated token to the sequence
            generated_seq = torch.cat([generated_seq, last_token_ids], dim=1)

            # Prepare the input for the next iteration
            tgt = torch.cat([tgt, last_token_ids], dim=1)

            # Stop if all sequences have produced an EOS token
            if not active_sequences.any():
                break

        # If some sequences are shorter than max_out_length, pad the rest with PAD tokens
        if generated_seq.size(1) < self.max_out_length:
            remaining_slots = self.max_out_length - generated_seq.size(1)
            padding = torch.full((batch_size, remaining_slots), PAD, dtype=torch.long, device=device)
            generated_seq = torch.cat([generated_seq, padding], dim=1)

        return generated_seq