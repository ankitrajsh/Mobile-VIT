import torchvision
import torchvision.transforms as transforms
import torch

####2. Load CIFAR-10 Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)





####3. Define the MobileViT Architecture


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility function for creating convolutional blocks
def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    layers = []
    layers.append(nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=bias))
    if norm:
        layers.append(nn.BatchNorm2d(oup))
    if act:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)

# Inverted Residual Block for MobileViT
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(conv_2d(inp, hidden_dim, kernel_size=1))
        layers.append(conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim))
        layers.append(conv_2d(hidden_dim, oup, kernel_size=1, norm=True, act=False))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

# Full MobileViT model setup
class MobileViT(nn.Module):
    def __init__(self, image_size, num_classes, mode='xx_small'):
        super().__init__()
        assert mode in ['xx_small', 'x_small', 'small'], "Mode should be one of 'xx_small', 'x_small', or 'small'"

        if mode == 'xx_small':
            layers_config = [
                # tuple format: (input_channels, output_channels, stride, expand_ratio)
                (3, 16, 1, 1),
                (16, 32, 2, 6),
                (32, 64, 2, 6),
                (64, 128, 2, 6),
                (128, 256, 2, 6),
                (256, 512, 2, 6)
            ]
        elif mode == 'x_small':
            layers_config = [
                (3, 16, 1, 1),
                (16, 48, 2, 6),
                (48, 96, 2, 6),
                (96, 192, 2, 6),
                (192, 384, 2, 6),
                (384, 768, 2, 6)
            ]
        elif mode == 'small':
            layers_config = [
                (3, 32, 1, 1),
                (32, 64, 2, 6),
                (64, 128, 2, 6),
                (128, 256, 2, 6),
                (256, 512, 2, 6),
                (512, 1024, 2, 6)
            ]

        layers = []
        for inp, oup, stride, expand_ratio in layers_config:
            layers.append(InvertedResidual(inp, oup, stride, expand_ratio))
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(layers_config[-1][1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Example: Instantiate a MobileViT model for CIFAR-10
num_classes = 10  # CIFAR-10 has 10 classes
image_size = (32, 32)  # CIFAR-10 images are 32x32
model = MobileViT(image_size, num_classes, mode='small')

print(model)









# ###4. Set Up Training directly for 10 epoch

# import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# for epoch in range(10):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')


################################################################################################################
################################################################################################################

###to monitor the training
# import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # Function to calculate accuracy
# def calculate_accuracy(outputs, labels):
#     _, predicted = torch.max(outputs, 1)
#     correct = (predicted == labels).sum().item()
#     return correct / labels.size(0)

# # Training loop
# for epoch in range(10):  # loop over the dataset multiple times
#     running_loss = 0.0
#     total_correct = 0
#     total_samples = 0
    
#     print(f"Epoch {epoch + 1}/{10} started...")
    
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         # Calculate metrics
#         running_loss += loss.item()
#         total_correct += (outputs.argmax(dim=1) == labels).sum().item()
#         total_samples += labels.size(0)
        
#         # Print status every 200 mini-batches
#         if (i + 1) % 200 == 0:
#             current_loss = running_loss / 200
#             accuracy = total_correct / total_samples * 100
#             print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {current_loss:.3f}, Accuracy: {accuracy:.2f}%")
#             running_loss = 0.0  # Reset running loss for the next interval

#     # End of epoch status
#     epoch_accuracy = total_correct / total_samples * 100
#     print(f"Epoch {epoch + 1} completed. Final Accuracy: {epoch_accuracy:.2f}%")

# print('Finished Training')

################################################################################################################
################################################################################################################

# ###tranig Training

# import torch.optim as optim
# from tqdm import tqdm  # For progress bar

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # Function to calculate accuracy
# def calculate_accuracy(outputs, labels):
#     _, predicted = torch.max(outputs, 1)
#     correct = (predicted == labels).sum().item()
#     return correct / labels.size(0)

# # Training loop
# num_epochs = 10  # Number of epochs
# for epoch in range(num_epochs):
#     model.train()  # Set model to training mode
#     running_loss = 0.0
#     total_correct = 0
#     total_samples = 0

#     # Display progress bar for the current epoch
#     print(f"\nEpoch {epoch + 1}/{num_epochs}")
#     progress_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc="Training")

#     for i, (inputs, labels) in progress_bar:
#         optimizer.zero_grad()  # Clear gradients from the previous step

#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         # Update metrics
#         running_loss += loss.item()
#         total_correct += (outputs.argmax(dim=1) == labels).sum().item()
#         total_samples += labels.size(0)

#         # Calculate average loss and accuracy so far
#         avg_loss = running_loss / (i + 1)
#         avg_accuracy = total_correct / total_samples * 100

#         # Update progress bar with batch metrics
#         progress_bar.set_postfix(
#             loss=f"{avg_loss:.4f}",
#             accuracy=f"{avg_accuracy:.2f}%",
#             processed=f"{total_samples}/{len(trainloader.dataset)}"
#         )

#     # Epoch summary
#     epoch_loss = running_loss / len(trainloader)
#     epoch_accuracy = total_correct / total_samples * 100
#     print(f"Epoch {epoch + 1} Summary: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")

# print("\nTraining Finished!")
################################################################################################################



################################################################################################################


# import time  # For tracking time
# import torch.optim as optim
# from tqdm import tqdm  # For progress bar

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # Function to calculate accuracy
# def calculate_accuracy(outputs, labels):
#     _, predicted = torch.max(outputs, 1)
#     correct = (predicted == labels).sum().item()
#     return correct / labels.size(0)

# # Training loop
# num_epochs = 10  # Number of epochs
# for epoch in range(num_epochs):
#     model.train()  # Set model to training mode
#     running_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#     epoch_start_time = time.time()  # Start time for the epoch

#     # Display progress bar for the current epoch
#     print(f"\nEpoch {epoch + 1}/{num_epochs}")
#     progress_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc="Training")

#     for i, (inputs, labels) in progress_bar:
#         batch_start_time = time.time()  # Start time for the batch

#         optimizer.zero_grad()  # Clear gradients from the previous step

#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         # Update metrics
#         running_loss += loss.item()
#         total_correct += (outputs.argmax(dim=1) == labels).sum().item()
#         total_samples += labels.size(0)

#         # Calculate average loss, accuracy, and time metrics
#         avg_loss = running_loss / (i + 1)
#         avg_accuracy = total_correct / total_samples * 100
#         time_per_iteration = time.time() - batch_start_time  # Time for this batch
#         elapsed_time = time.time() - epoch_start_time
#         remaining_time = elapsed_time / (i + 1) * (len(trainloader) - (i + 1))

#         # Update progress bar with batch metrics
#         progress_bar.set_postfix(
#             loss=f"{avg_loss:.4f}",
#             accuracy=f"{avg_accuracy:.2f}%",
#             time_per_iter=f"{time_per_iteration:.2f}s",
#             elapsed=f"{elapsed_time:.2f}s",
#             remaining=f"{remaining_time:.2f}s",
#         )

#     # Epoch summary
#     epoch_loss = running_loss / len(trainloader)
#     epoch_accuracy = total_correct / total_samples * 100
#     epoch_end_time = time.time()
#     epoch_duration = epoch_end_time - epoch_start_time
#     print(f"Epoch {epoch + 1} Summary: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%, Time = {epoch_duration:.2f}s")

# print("\nTraining Finished!")
################################################################################################################

##################################################################################################################
###hyperparameter details 
"""
Category             Hyperparameter        Value
---------------------------------------------------------
Model                Model Type            MobileViT (xx_small)
                     Image Size            (32, 32)
                     Number of Classes     10
Dataset/DataLoader   Batch Size            64
                     Shuffle               True (for training)
                     Num Workers           2
Optimization         Learning Rate         0.001
                     Momentum              0.9
Loss Function        Criterion             CrossEntropyLoss
Training             Number of Epochs      10
"""


import time
import torch
import torch.optim as optim
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Assuming MobileViT is already defined
# Prepare dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Mean and standard deviation for each channel
])
###batch size 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Define model using xx_small
image_size = (32, 32)
num_classes = 10
model = MobileViT(image_size=image_size, num_classes=num_classes, mode='xx_small')

# Move model to GPU
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    epoch_start_time = time.time()

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    progress_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc="Training")

    for i, (inputs, labels) in progress_bar:
        batch_start_time = time.time()

        # Move data to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

        avg_loss = running_loss / (i + 1)
        avg_accuracy = total_correct / total_samples * 100
        time_per_iteration = time.time() - batch_start_time
        elapsed_time = time.time() - epoch_start_time
        remaining_time = elapsed_time / (i + 1) * (len(trainloader) - (i + 1))

        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            accuracy=f"{avg_accuracy:.2f}%",
            time_per_iter=f"{time_per_iteration:.2f}s",
            elapsed=f"{elapsed_time:.2f}s",
            remaining=f"{remaining_time:.2f}s"
        )

    epoch_loss = running_loss / len(trainloader)
    epoch_accuracy = total_correct / total_samples * 100
    print(f"Epoch {epoch + 1} Summary: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%, Time = {elapsed_time:.2f}s")

print("\nTraining Finished!")

################################################################################################################








###5. Test the Model

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
