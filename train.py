import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

# Command line inputs
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('data_directory', help='Basic usage')
parser.add_argument('--save_dir', help='Set directory to save checkpoints')
parser.add_argument('--arch', help='Choose architecture')
parser.add_argument('--hidden_units', help='Set hyperparameters -  hidden units')
parser.add_argument('--learning_rate', help='Set hyperparameters - learning_rate')
parser.add_argument('--epochs', help='Set hyperparameters - epochs')
parser.add_argument('--gpu', help='Use GPU for training', action='store_true')
args_input = parser.parse_args()

# Define default inputs
arch = args_input.arch if args_input.arch else 'vgg11'
hidden_units = int(args_input.hidden_units) if args_input.hidden_units else 256
lr = float(args_input.learning_rate) if args_input.learning_rate else 0.001
epochs = int(args_input.epochs) if args_input.epochs else 10
#  ------------------------ Training -----------------------------------

# Inputs path
data_dir = args_input.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

print('Inputs path: ', data_dir)

print('-------------------------- Data loading and batching --------------------------')
# Data augmentation and Data normalization
data_transforms_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(degrees=25),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Compose
data_transforms_valid = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_test = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data loading
train_image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms_train)
valid_image_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)
test_image_datasets = datasets.ImageFolder(test_dir, transform=data_transforms_test)

# Data batching
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64, shuffle=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=64, shuffle=False)

print('-------------------------- Define architecture --------------------------')
# Define pretrained network
model = models.vgg11(pretrained=True) if arch == 'vgg11' \
    else models.vgg16(pretrained=True) if arch == 'vgg16' \
    else models.vgg19(pretrained=True) if arch == 'vgg19' \
    else models.vgg11(pretrained=True)

print('Architecture selected: ', arch)

print('-------------------------- Froze parameters --------------------------')
# Parameters are frozen
for param in model.features.parameters():
    param.requires_grad = False
print('-------------------------- Define classifier --------------------------')

# Classifier
Classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.4),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim=1)
                           )

# Training the network
model.classifier = Classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args_input.gpu else torch.device("cpu")
model.to(device)
print('Device selected: ', device)

print_step = 12
steps = 0
train_loss = 0

# Training process
print('-------------------------- Training --------------------------')
for epoch in range(epochs):
    for inputs, labels in train_dataloaders:
        inputs, labels = inputs.to(device), labels.to(device)
        steps += 1

        optimizer.zero_grad()

        # Forward
        output = model.forward(inputs)

        # Loss
        loss = criterion(output, labels)

        # Backrard
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Validation Loss and Accuracy
        if steps % print_step == 0:
            val_loss = 0
            val_accuracy = 0

            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_dataloaders:
                    inputs, labels = inputs.to(device), labels.to(device)

                    output = model.forward(inputs)
                    b_loss = criterion(output, labels)
                    val_loss += b_loss.item()

                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {train_loss / print_step:.3f}.. "
                  f"Val loss: {val_loss / len(valid_dataloaders):.3f}.. "
                  f"Val accuracy: {val_accuracy / len(valid_dataloaders):.3f}")

            train_loss = 0
            model.train()

# Testing
print('-------------------------- Test --------------------------')
model.to(device)
test_loss = 0
test_accuracy = 0

model.eval()
with torch.no_grad():
    for inputs, labels in test_dataloaders:
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        b_loss = criterion(output, labels)
        test_loss += b_loss.item()

        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test loss: {test_loss / len(test_dataloaders):.3f}.. "
      f"Test accuracy: {test_accuracy / len(test_dataloaders):.3f}")

# Save checkpoint
print('-------------------------- Save checkpoint --------------------------')
model.class_to_idx = train_image_datasets.class_to_idx
checkpoint = {
    'arch': arch,
    'hidden_units': hidden_units,
    'class_to_idx': model.class_to_idx,
    'm_state_dict': model.state_dict()}

torch.save(checkpoint, args_input.save_dir + 'checkpoint2.pth' if args_input.save_dir else '')
# print(args_input.save_dir + 'checkpoint.pth' if args_input.save_dir else '')
print('-------------------------- End --------------------------')
