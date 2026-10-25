import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from .mixup import MixUpCollator

def get_food101n_dataloader(args):
    train_dir = os.path.join(args.data_dir, 'Food101N', 'train')
    test_dir = os.path.join(args.data_dir, 'Food101N', 'val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    basic_transform = [transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip()]
        
    basic_transform += [transforms.ToTensor(), normalize]
        
    train_transform = transforms.Compose(basic_transform)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    full_train_set = datasets.ImageFolder(train_dir, train_transform)
    train_ratio = 0.9

    # Calculate the number of samples for each set
    num_total_samples = len(full_train_set)
    num_train_samples = int(train_ratio * num_total_samples)
    num_val_samples = num_total_samples - num_train_samples
    
    generator = torch.Generator().manual_seed(42) # Use any seed value

    train_subset, val_subset = torch.utils.data.random_split(
        full_train_set,
        [num_train_samples, num_val_samples],
        generator=generator
    )
    
    test_set = datasets.ImageFolder(test_dir, test_transform)
        
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=args.batch_size_train,
        num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=args.batch_size_eval,
        num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size_eval,
        num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False
    )
    
    return train_loader, test_loader, val_loader