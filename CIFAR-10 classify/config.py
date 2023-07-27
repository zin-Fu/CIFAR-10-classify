import torch
# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10

# Architecture
NUM_FEATURES = 32 * 32
NUM_CLASSES = 10

# Other
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAYSCALE = False
