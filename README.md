---
## Authors

Christopher Mountain, Vatsal Bagri, Sameer Bharatia

# Image to Latex
## Description

This project aims to implement a ResNet (encoder) + Transfomer (decoder) model that converts images to LaTeX code. This is particularly useful for automating the digitization of handwritten or typeset mathematical documents into editable text formats. The repository includes scripts for preprocessing data, defining the neural network model architecture, training the model, and evaluating its performance. The training process leverages multiple GPUs to handle large datasets efficiently.

## File Structure

- `dataset.py`: Manages the loading and preprocessing of image and LaTeX data for training.
- `main.py`: The entry point for running the complete training process, including setup and execution control.
- `model.py`: Defines the neural network architecture used for the image to LaTeX translation task.
- `training.py`: Contains the core training logic, including backpropagation and validation.
- `training_metrics.csv`: A CSV file that logs the loss and BLEU score metrics after each training epoch.
- `training_output.txt`: Text file containing detailed output of the training process, including per-epoch metrics and final test results.

## Getting Started

### Dependencies

- Python 3.8+
- TensorFlow 2.x or PyTorch 1.x
- CUDA (for GPU support)

### Executing Program

```bash
python main.py
```


---
