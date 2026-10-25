import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from .mixup import MixUpCollator

def get_imagenet_dataset(args):
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'val')
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
    
    train_set = datasets.ImageFolder(train_dir, train_transform)
    test_set = datasets.ImageFolder(test_dir, test_transform)
        
    if args.use_mixup:
        collator = MixUpCollator(args.mixup_alpha)
    else:
        collator = torch.utils.data.dataloader.default_collate
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size_train,
        num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collator
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size_eval,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
    )
    
    return train_loader, test_loader