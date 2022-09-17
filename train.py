import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchmetrics.functional.classification.accuracy import _subset_accuracy_update as correct_total
import torchvision.transforms as T
from torchvision.datasets.folder import pil_loader

def get_args():
    parser = argparse.ArgumentParser()
    # Dataset args
    parser.add_argument('--root', type=str, default=r"C:\Users\pennstateuas\projects\datasets\fgvc-aircraft-2013b-128")
    parser.add_argument('--input_size', default=64, type=int)
    # Dataloader args
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--pin_memory', default=False, action='store_true')
    # Model args
    parser.add_argument('--features', default=16, type=int)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    # Optimizer args
    parser.add_argument('--lr', default=4e-3, type=float)
    # Scheduler args
    # Training
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--max_steps', default=None, type=int)
    parser.add_argument('--val_every_n_steps', default=10, type=int)
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    args = parser.parse_args()
    return args

def get_label_filename(root, split):
    return f"{root}\images_family_{split}.txt"

class AircraftDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        super().__init__()
        self.root = root
        self.split = split
        self.label_filename = get_label_filename(root, split)
        def make_dict_item(line):
            """ Turn one line into a dictionary. """
            base, label = line.strip("\n").split(" ", 1)
            return {"image_path": f"{self.root}\images\{base}.jpg", "label": label}
        # Load the label file and convert to a list of dictionarys
        with open(self.label_filename, "r") as filehandle:
            self.data_dict = [make_dict_item(l) for l in filehandle.readlines()]
        # Map the labels to an integer
        self.labels = list(set([e["label"] for e in self.data_dict]))
        self.num_classes = len(self.labels)
        self.label_stoi = {k:i for i,k in enumerate(self.labels)}
        self.label_itos = {v:k for k,v in self.label_stoi.items()}
        # Default transform converts the pil image to a tensor
        self.transforms = T.Compose([T.ToTensor()])
        if transforms:
            self.transforms = transforms

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        img_ten = self.transforms(pil_loader(data["image_path"]))
        label_id = self.label_stoi[data["label"]]
        return img_ten, label_id

# Model Class
def quantize_to_int(x, q=8):
    """ Make interger divisible by q, but never smaller than q. """    
    return int(q) if x<q else int(np.floor(x/q)*q)

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bottleneck_ratio=0,
                activation=None, norm=nn.BatchNorm2d, dropout=0.0):
        super(ResidualLayer, self).__init__()
        self.pre_activation = nn.Sequential()
        if norm: self.pre_activation.add_module("norm", norm(in_channels))
        if activation: self.pre_activation.add_module("act", activation())

        self.residual = nn.Sequential()
        if bottleneck_ratio>0:
            mid_channels = quantize_to_int(out_channels*bottleneck_ratio, 8)
            self.residual.add_module("conv1", nn.Conv2d(in_channels, mid_channels,
                    kernel_size=(1,1), stride=1, padding=0, bias=False))
            if norm: self.residual.add_module("norm1", norm(mid_channels))
            if activation: self.residual.add_module("act1", activation())
            self.residual.add_module("drop1", nn.Dropout(dropout))
            self.residual.add_module("conv2", nn.Conv2d(mid_channels, mid_channels,
                    kernel_size=(3,3), stride=stride, padding=1, bias=False))
            if norm: self.residual.add_module("norm2", norm(mid_channels))
            if activation: self.residual.add_module("act2", activation())
            self.residual.add_module("drop2", nn.Dropout(dropout))
            self.residual.add_module("conv3", nn.Conv2d(mid_channels, out_channels,
                    kernel_size=(1,1), stride=1, padding=0, bias=False))
        else:
            self.residual.add_module("conv1", nn.Conv2d(in_channels, out_channels,
                    kernel_size=(3,3), stride=stride, padding=1, bias=False))
            if norm: self.residual.add_module("norm1", norm(out_channels))
            if activation: self.residual.add_module("act", activation())
            self.residual.add_module("drop1", nn.Dropout(dropout))
            self.residual.add_module("conv2", nn.Conv2d(out_channels, out_channels,
                    kernel_size=(3,3), stride=1, padding=1, bias=False))

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module("downsample", nn.AvgPool2d(kernel_size=stride, stride=stride))
            self.shortcut.add_module("conv", nn.Conv2d(in_channels, out_channels,
                    kernel_size=(1,1), stride=1, bias=False))
        else:
            self.shortcut = None

    def forward(self, x):
        activated = self.pre_activation(x)
        shortcut = x
        if self.shortcut is not None:
            shortcut = self.shortcut(activated)
        residual = self.residual(activated)
        return shortcut + residual

class SmallModel(nn.Module):
    def __init__(self, in_channels, features, classes, patch_size=4,
                activation=nn.SiLU, norm=nn.BatchNorm2d, dropout=0.0):
        super().__init__()
        features = [features, features*2, features*4, features*4, features*8]
        self.patch_embed = nn.Conv2d(in_channels, features[0], kernel_size=patch_size, stride=patch_size)
        self.features = nn.Sequential(
            self.residual_block(blocks=2, in_channels=features[0], out_channels=features[1], stride=1, bottleneck_ratio=0, activation=activation, norm=norm, dropout=dropout),
            self.residual_block(blocks=2, in_channels=features[1], out_channels=features[2], stride=2, bottleneck_ratio=0, activation=activation, norm=norm, dropout=dropout),
            self.residual_block(blocks=2, in_channels=features[2], out_channels=features[3], stride=2, bottleneck_ratio=0, activation=activation, norm=norm, dropout=dropout),
            self.residual_block(blocks=2, in_channels=features[3], out_channels=features[4], stride=2, bottleneck_ratio=0, activation=activation, norm=norm, dropout=dropout),
        )
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.class_proj = nn.Linear(features[4], classes)

    def residual_block(self, blocks, in_channels, out_channels, stride, bottleneck_ratio, activation, norm, dropout):
        layers = [ResidualLayer(in_channels, out_channels, stride, bottleneck_ratio, activation, norm, dropout)]
        for idx in range(1, blocks):  # Downsample with stride on the first block only
            layers.append(ResidualLayer(out_channels, out_channels, 1, bottleneck_ratio, activation, norm, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        patches = self.patch_embed(x)
        features = self.features(patches)
        embedding = self.flatten(features)
        logits = self.class_proj(embedding)
        return logits

def process_batch(batch, model, loss_fn, no_grad=False):
    images, labels = batch
    images = images.to(model.patch_embed.weight.device)
    labels = labels.to(model.patch_embed.weight.device)
    yhat = model(images)
    yhat = yhat.detach() if no_grad else yhat
    loss = loss_fn(yhat, labels)
    correct, seen = correct_total(yhat, labels, threshold=0.5, top_k=1)
    metrics = {
        "loss": loss,
        "seen": seen.item(),
        "correct": correct.item(),
        "acc": (correct / seen).item(),
    }
    return metrics

def list_of_dict_to_stats(list_of_dicts):
    avg_stats = {}
    for k in list_of_dicts[0].keys():
        avg_stats[k] = sum(d[k] for d in list_of_dicts)
        if k not in ["seen", "correct"]:
            avg_stats[k] = avg_stats[k]/len(list_of_dicts)
    return avg_stats

def main():
    args = get_args()

    # Create the datasets using these transforms
    valid_transform = T.Compose([
        T.Resize(args.input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
    ])
    train_transform = T.Compose([
        T.RandomResizedCrop(args.input_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5, hue=0.1),
        T.RandomHorizontalFlip(),
        T.RandomGrayscale(p=0.25),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
    ])

    train_ds = AircraftDataset(args.root, split="train", transforms=train_transform)  # For training
    val_ds = AircraftDataset(args.root, split="val", transforms=valid_transform)  # For validation while training
    test_ds = AircraftDataset(args.root, split="test", transforms=valid_transform)  # For evaluation after training

    # Create the dataloaders for each dataset split
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch, shuffle=True,
            num_workers=args.workers, persistent_workers=(True if args.workers > 0 else False),
            pin_memory=args.pin_memory)
    val_loader = DataLoader(dataset=val_ds, batch_size=args.batch, shuffle=False,
            num_workers=args.workers, persistent_workers=(True if args.workers > 0 else False),
            pin_memory=args.pin_memory)
    test_loader = DataLoader(dataset=test_ds, batch_size=args.batch, shuffle=False,
            num_workers=args.workers, persistent_workers=(True if args.workers > 0 else False),
            pin_memory=args.pin_memory)

    # Check for gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # Load model, optimizer, scheduler
    model = SmallModel(in_channels=3, features=args.features,
                    classes=train_ds.num_classes, patch_size=args.patch_size,
                    dropout=args.dropout
                ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None

    # Create the loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Check if you should stop training by steps or epoch
    stop_training = lambda s, e: (args.max_steps and s>=args.max_steps) or (args.max_epochs and e>args.max_epochs)

    steps = 0  # Total number of update steps
    print(" * Start training.")
    for epoch in range(1, args.max_epochs+1):  # Start epochs at 1
        # Check if this is the final epoch
        if stop_training(steps, epoch): break

        # Loop over all the images in the train dataloader
        train_metrics = []
        for batch in train_loader:
            steps += 1

            # Train batch with gradients
            metrics = process_batch(batch, model, loss_fn, no_grad=False)
            train_metrics.append(metrics)
            # Update the weights of the model
            optimizer.zero_grad()
            metrics["loss"].backward()
            optimizer.step()

            # Check if this is the final step
            if stop_training(steps, epoch): break
            
            # Validation loop val_every_n_steps
            if steps%args.val_every_n_steps==0:
                val_metrics = []
                for batch in val_loader:
                    metrics = process_batch(batch, model, loss_fn, no_grad=True)
                    val_metrics.append(metrics)
                val_stats = list_of_dict_to_stats(val_metrics)
                print(f"VALIDATINON {epoch=} {steps=}", val_stats)

        # End train epoch, report the stats
        train_stats = list_of_dict_to_stats(train_metrics)
        print(f"TRAINING {epoch=} {steps=}", train_stats)

    print(" * End training")

    # Save the model weights

    # Test the model on the evaluation data
    print(f"Run evaluation. {epoch=} {steps=}")
    eval_metrics = []
    for batch in val_loader:
        metrics = process_batch(batch, model, loss_fn, no_grad=True)
        eval_metrics.append(metrics)
    eval_stats = list_of_dict_to_stats(eval_metrics)
    print("EVALUATION", eval_stats)


if __name__ == "__main__":
    main()
