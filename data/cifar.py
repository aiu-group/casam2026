import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np

from .cutout import Cutout
from .cutmix import CutMixCollator
from .mixup import MixUpCollator


# https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
class CIFAR10RandomLabels(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.

    Params
    ------
    corrupt_prob: float
      Default 0.0. The probability of a label being replaced with
      random label.
    num_classes: int
      Default 10. The number of classes in the dataset.
    """

    def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        self.targets = labels


class CIFAR100RandomLabels(datasets.CIFAR100):
    """CIFAR100 dataset, with support for randomly corrupt labels.

    Params
    ------
    corrupt_prob: float
      Default 0.0. The probability of a label being replaced with
      random label.
    num_classes: int
      Default 100. The number of classes in the dataset.
    """

    def __init__(self, corrupt_prob=0.0, num_classes=100, **kwargs):
        super(CIFAR100RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        self.targets = labels


def get_cifar_dataloader(args):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # basic_transform = [transforms.RandomCrop(32, padding=4),
    #                 transforms.RandomHorizontalFlip(),
    #                 ]
    basic_transform = [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze()
        ),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
    ]

    if args.use_auto_augment:
        basic_transform.append(
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
        )
    if args.use_rand_augment:
        basic_transform.append(transforms.RandAugment(num_ops=2, magnitude=9))
    if args.use_random_erasing:
        basic_transform.append(transforms.RandomErasing(p=0.25))

    basic_transform += [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        normalize,
    ]
    if args.use_cutout:
        basic_transform.append(Cutout(n_holes=1, length=args.length))

    train_transform = transforms.Compose(basic_transform)
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if args.use_cutmix:
        collator = CutMixCollator(args.cutmix_alpha)
    elif args.use_mixup:
        collator = MixUpCollator(args.mixup_alpha)
    else:
        collator = torch.utils.data.dataloader.default_collate

    if args.data_name == "cifar10":
        # load the training set twice because data augmentations on training set and validation set may be different
        classes = 10
        valid_per_class = 500
        train_dataset = CIFAR10RandomLabels(
            root=args.data_dir,
            train=True,
            transform=train_transform,
            download=True,
            corrupt_prob=args.corrupt_prob,
        )
        val_dataset = CIFAR10RandomLabels(
            root=args.data_dir,
            train=True,
            transform=test_transform,
            download=True,
            corrupt_prob=0.0,
        )
        # val_dataset = torchvision.datasets.CIFAR10(args.data_dir, train=True, transform=test_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            args.data_dir, train=False, transform=test_transform, download=True
        )
    elif args.data_name == "cifar100":
        classes = 100
        valid_per_class = 50
        train_dataset = CIFAR100RandomLabels(
            root=args.data_dir,
            train=True,
            transform=train_transform,
            download=True,
            corrupt_prob=args.corrupt_prob,
        )
        val_dataset = CIFAR100RandomLabels(
            root=args.data_dir,
            train=True,
            transform=test_transform,
            download=True,
            corrupt_prob=0.0,
        )
        # val_dataset = torchvision.datasets.CIFAR100(args.data_dir, train=True, transform=test_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            args.data_dir, train=False, transform=test_transform, download=True
        )
    else:
        raise NotImplementedError(f"{args.data_name} is not avaiable right now !")

    torch.manual_seed(43)  # ensure the validation set is the same for each run
    indices = torch.randperm(len(train_dataset))
    val_size = 5000
    valid_indices, train_indices = indices[:val_size], indices[val_size:]
    # targets = np.array(val_dataset.targets)
    # valid_indices = []
    # train_indices = []
    # for class_label in range(classes):
    #     class_indices = np.where(targets == class_label)[0]
    #     valid_indices.append(class_indices[:valid_per_class])
    #     train_indices.append(class_indices[valid_per_class:])
    # valid_indices = np.hstack(valid_indices)
    # train_indices = np.hstack(train_indices)

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, valid_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        prefetch_factor=2,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_eval,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
        drop_last=False,
        prefetch_factor=2,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        prefetch_factor=2,
    )
    return train_loader, test_loader, val_loader
