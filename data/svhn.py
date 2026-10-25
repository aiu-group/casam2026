import os
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .mixup import MixUpCollator


def get_svhn_train_loader(
    args,
    random_seed=1314,
    valid_size=0.1,
    shuffle=True,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the SVHN dataset.
    Params:
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    MEAN = [0.49139968, 0.48215841, 0.44653091]
    STD = [0.24703223, 0.24348513, 0.26158784]

    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    # define transforms
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    # if augment:
    #    train_transform = transforms.Compose([
    #        transforms.RandomCrop(32, padding=4),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        normalize,
    #    ])
    # else:
    #    train_transform = transforms.Compose([
    #        transforms.ToTensor(),
    #        normalize,
    #    ])

    # load the dataset
    data_dir = args.data_dir
    train_dataset = datasets.SVHN(
        root=data_dir,
        split="train",
        download=True,
        transform=valid_transform,
    )

    if args.use_mixup:
        collator = MixUpCollator(args.mixup_alpha)
    else:
        collator = torch.utils.data.dataloader.default_collate

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collator,
    )

    return train_loader


def get_svhn_test_loader(args, shuffle=False, random_seed=1314):
    """
    Utility function for loading and returning a multi-process
    test iterator over the SVHN dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    MEAN = [0.49139968, 0.48215841, 0.44653091]
    STD = [0.24703223, 0.24348513, 0.26158784]

    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    # define transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    data_dir = args.data_dir
    dataset = datasets.SVHN(
        root=data_dir,
        split="test",
        download=True,
        transform=transform,
    )

    # valid_dataset = datasets.SVHN(
    #     root=data_dir, split='test',
    #     download=True, transform=transform,
    # )

    # num_test = len(dataset)
    # indices = list(range(num_test))
    # split = int(np.floor(0.2 * num_test))

    # if shuffle:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)

    # test_idx, valid_idx = indices[split:], indices[:split]
    # test_sampler = SubsetRandomSampler(test_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    # test_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=args.batch_size_eval, sampler=test_sampler,
    #     num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False
    # )

    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=args.batch_size_eval, sampler=valid_sampler,
    #     num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False
    # )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_eval,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
    )

    return test_loader
