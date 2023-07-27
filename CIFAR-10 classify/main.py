import argparse
from model import *
from train import *
from val import *

device = torch.device(DEVICE)
torch.manual_seed(RANDOM_SEED)

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--model', type=str, default='Lenet5', choices=['Lenet5', 'Resnet'],
                    help='Model name (default: Lenet5)')

print("🚀 Building model...")
args = parser.parse_args()
if args.model == 'Lenet5':
    model = LeNet5(num_classes=NUM_CLASSES, grayscale=GRAYSCALE)
    print("--------Model built: LeNet5\n")
elif args.model == 'Resnet':
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=NUM_CLASSES, grayscale=GRAYSCALE)
    print("--------Model built: ResNet\n")
model.to(DEVICE)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("🚀Training on ", device)
train(model=model, optimizer=optimizer)
print("Training completed!\n")

print("Evaluating model...\n")
evaluation_and_show(model=model)