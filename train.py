import argparse
import time

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
    # Optimizer args
    # Scheduler args
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
        self.label_stoi = {k:i+1 for i,k in enumerate(self.labels)}
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

    # DEBUG Sample a test batch from the train loader to confirm shapes
    x, y = next(iter(train_loader))
    print(type(x), x.device, x.type, x.shape)
    print(type(y), y.device, y.dtype, y.shape)

    # Check for gpu
    # Load model, optimizer, scheduler
    # Loop
    # Train for val_every_n_steps
    # Run validation loop and resume training
    # end training after train_steps
    # Save the model weights
    # Run the test loop
    # Save the args and the val/test scores

if __name__ == "__main__":
    main()
