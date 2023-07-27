# CIFAR-10-classify

This is a training script to train image classification models on a dataset. The script allows for the training of two different models: LeNet5 and ResNet.

### Dependencies
Python 3
PyTorch
argparse
numpy
Command Line Arguments
--model: Specifies the model to use. The available options are LeNet5 and ResNet, with LeNet5 being the default.
### Usage
Install all dependencies.
Run the following command in the terminal: python train.py
Optionally, use the --model argument to specify the model to use.
Output
For each training and evaluation process, the script outputs relevant information such as the model architecture, training loss and accuracy, etc.

Upon completion of the training process, the script outputs the model's accuracy on the validation set and displays some randomly selected images along with their predicted and true labels.
