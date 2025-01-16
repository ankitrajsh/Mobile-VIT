import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F




def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        # hidden_dim = int(round(inp * expand_ratio))
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0))
        self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, act=False))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)  


class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim, attn_dropout=0):
        super().__init__()
        self.qkv_proj = conv_2d(embed_dim, 1+2*embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = conv_2d(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.embed_dim = embed_dim

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)
        context_score = F.softmax(q, dim=-1)
        context_score = self.attn_dropout(context_score)

        context_vector = k * context_score
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(v) * context_vector.expand_as(v)
        out = self.out_proj(out)
        return out

class LinearAttnFFN(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, dropout=0, attn_dropout=0):
        super().__init__()
        self.pre_norm_attn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            LinearSelfAttention(embed_dim, attn_dropout),
            nn.Dropout(dropout)
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            conv_2d(embed_dim, ffn_latent_dim, kernel_size=1, stride=1, bias=True, norm=False, act=True),
            nn.Dropout(dropout),
            conv_2d(ffn_latent_dim, embed_dim, kernel_size=1, stride=1, bias=True, norm=False, act=False),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # self attention
        x = x + self.pre_norm_attn(x)
        # Feed Forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlockv2(nn.Module):
    def __init__(self, inp, attn_dim, ffn_multiplier, attn_blocks, patch_size):
        super(MobileViTBlockv2, self).__init__()
        self.patch_h, self.patch_w = patch_size

        # local representation
        self.local_rep = nn.Sequential()
        self.local_rep.add_module('conv_3x3', conv_2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp))
        self.local_rep.add_module('conv_1x1', conv_2d(inp, attn_dim, kernel_size=1, stride=1, norm=False, act=False))
        
        # global representation
        self.global_rep = nn.Sequential()
        ffn_dims = [int((ffn_multiplier*attn_dim)//16*16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep.add_module(f'LinearAttnFFN_{i}', LinearAttnFFN(attn_dim, ffn_dim))
        self.global_rep.add_module('LayerNorm2D', nn.GroupNorm(num_channels=attn_dim, eps=1e-5, affine=True, num_groups=1))

        self.conv_proj = conv_2d(attn_dim, inp, kernel_size=1, stride=1, padding=0, act=False)

    def unfolding_pytorch(self, feature_map):
        batch_size, in_channels, img_h, img_w = feature_map.shape
        if img_h < self.patch_h or img_w < self.patch_w:
            raise ValueError(
                f"Input dimensions are too small for patch size. "
                f"Input: ({img_h}, {img_w}), Patch: ({self.patch_h}, {self.patch_w})"
            )
        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)


    def folding_pytorch(self, patches, output_size):
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def forward(self, x):
        print(f"Input shape to MobileViTBlockv2: {x.shape}")
        x = self.local_rep(x)
        print(f"After local_rep: {x.shape}")
        x, output_size = self.unfolding_pytorch(x)
        print(f"After unfolding: {x.shape}")
        x = self.global_rep(x)
        print(f"After global_rep: {x.shape}")
        x = self.folding_pytorch(patches=x, output_size=output_size)
        print(f"After folding: {x.shape}")
        x = self.conv_proj(x)
        print(f"After conv_proj: {x.shape}")
        return x



class MobileViTv2(nn.Module):
    def __init__(self, image_size, width_multiplier, num_classes, patch_size=(2, 2)):  
        super().__init__()
        # check image size
        ih, iw = image_size
        self.ph, self.pw = patch_size
        assert ih % self.ph == 0 and iw % self.pw == 0 
        assert width_multiplier in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

        # model size
        channels = []
        channels.append(int(max(16, min(64, 32 * width_multiplier))))
        channels.append(int(64 * width_multiplier))
        channels.append(int(128 * width_multiplier))
        channels.append(int(256 * width_multiplier))
        channels.append(int(384 * width_multiplier))
        channels.append(int(512 * width_multiplier))
        attn_dim = []
        attn_dim.append(int(128 * width_multiplier))
        attn_dim.append(int(192 * width_multiplier))
        attn_dim.append(int(256 * width_multiplier))

        # default shown in paper
        ffn_multiplier = 2
        mv2_exp_mult = 2

        self.conv_0 = conv_2d(3, channels[0], kernel_size=3, stride=2)

        self.layer_1 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.layer_2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mv2_exp_mult),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.layer_3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockv2(channels[3], attn_dim[0], ffn_multiplier, 2, patch_size=patch_size)
        )
        self.layer_4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockv2(channels[4], attn_dim[1], ffn_multiplier, 4, patch_size=patch_size)
        )
        self.layer_5 = nn.Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockv2(channels[5], attn_dim[2], ffn_multiplier, 3, patch_size=patch_size)
        )
        self.out = nn.Linear(channels[-1], num_classes, bias=True)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x) 
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        
        # FF head
        x = torch.mean(x, dim=[-2, -1])
        x = self.out(x)

        return x

# Define transformations for CIFAR
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])


# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

"""
Category             Hyperparameter        Value
---------------------------------------------------------
Model                Model Type            MobileViT (v2)
                     Width Multiplier      1.0
                     Patch Size            (2, 2)
                     FFN Multiplier        2
                     MV2 Expansion Mult.   2
                     Attention Blocks      [2, 4, 3]
                     Image Size            (32, 32)
                     Number of Classes     10

Dataset/DataLoader   Batch Size            128
                     Shuffle               True (for training)
                     Num Workers           2
                     Random Crop Padding   4
                     Random Flip Prob.     0.5
                     Normalization Mean    (0.4914, 0.4822, 0.4465)
                     Normalization Std     (0.247, 0.243, 0.261)

Optimization         Learning Rate         0.001
                     Optimizer             AdamW
                     Weight Decay          1e-4
                     Scheduler             CosineAnnealingLR

Loss Function        Criterion             CrossEntropyLoss

Training             Number of Epochs      200

Hardware             Device                CUDA (if available)
"""


# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileViTv2(image_size=(32, 32), width_multiplier=1.0, num_classes=10, patch_size=(2, 2))
model.to(device)



# Define optimizer, loss, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)






# Training function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    scheduler.step()

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

print('Training complete.')



####################################################################################################

import time

# Training function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_duration = time.time() - start_time
    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    iterations_per_second = len(loader) / epoch_duration
    return epoch_loss, accuracy, epoch_duration, iterations_per_second

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_duration = time.time() - start_time
    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    iterations_per_second = len(loader) / epoch_duration
    return epoch_loss, accuracy, epoch_duration, iterations_per_second

# Training loop
num_epochs = 200
train_dataset_size = len(train_loader.dataset)
val_dataset_size = len(test_loader.dataset)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}:")
    train_loss, train_acc, train_time, train_iters_per_sec = train_epoch(
        model, train_loader, optimizer, criterion, device
    )
    val_loss, val_acc, val_time, val_iters_per_sec = validate(
        model, test_loader, criterion, device
    )
    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Train Time: {train_time:.2f}s, Iter/s: {train_iters_per_sec:.2f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Val Time: {val_time:.2f}s, Iter/s: {val_iters_per_sec:.2f}")
    print(f"Total Training Samples: {train_dataset_size}, Validation Samples: {val_dataset_size}")
    print("-" * 50)

print("Training complete.")
print(inputs.shape)
