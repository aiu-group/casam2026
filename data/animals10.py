import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import os


def get_animal10_dataloader(args):    

    # transformation of the training set
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # transformation of the validation set
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    animal10n_full_train_dataset = ImageFolder(os.path.join(args.data_dir, 'animals-10', 'train'), transform_train)
    
    train_ratio = 0.9

    # Calculate the number of samples for each set
    num_total_samples = len(animal10n_full_train_dataset)
    num_train_samples = int(train_ratio * num_total_samples)
    num_val_samples = num_total_samples - num_train_samples
    
    generator = torch.Generator().manual_seed(42) # Use any seed value

    train_subset, val_subset = random_split(
        animal10n_full_train_dataset,
        [num_train_samples, num_val_samples],
        generator=generator
    )

    train_loader = DataLoader(train_subset,
                                batch_size=args.batch_size_train, 
                                shuffle=True, 
                                drop_last=False, 
                                num_workers = 4, 
                                pin_memory = True)
    
    val_loader = DataLoader(val_subset,
                                batch_size=args.batch_size_eval, 
                                shuffle=False, 
                                drop_last=False, 
                                num_workers = 4, 
                                pin_memory = True)
    
    test_loader = DataLoader(ImageFolder(os.path.join(args.data_dir, 'animals-10', 'test'), transform_test),
                            batch_size=args.batch_size_eval, 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers = 4, 
                            pin_memory = True)

                                
    return  train_loader, test_loader, val_loader