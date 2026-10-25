import random
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join
import argparse
from tqdm import tqdm
from colorama import Fore, Style
import wandb

import json
from prefetch_generator import BackgroundGenerator
from copy import deepcopy

from data import SoftCrossEntropyLoss
from data import get_cifar_dataloader

from pytorchcv.model_provider import get_model as ptcv_get_model
from opt import SAM

from utils import GammaScheduler, bern_kl_div

# avaiable neural architectures
model_zoo = [
    "resnet20_cifar10",
    "resnet20_cifar100",
    "resnet56_cifar10",
    "resnet56_cifar100",
    "resnext29_32x4d_cifar10",
    "resnext29_32x4d_cifar100",
    "wrn28_10_cifar10",
    "wrn28_10_cifar100",
    "pyramidnet110_a270_cifar10",
    "pyramidnet110_a270_cifar100",
]


def init_random_state(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def create_net(args):
    init_random_state(args.seed)
    if args.arch in model_zoo:
        net = ptcv_get_model(args.arch, pretrained=False)
    else:
        raise NotImplementedError(f"{args.arch} not implemented")
    return net


def normalize_func(tensor):
    # Avoid division by zero in case max == min
    if tensor.max() == tensor.min():
        return torch.zeros_like(tensor)

    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def hyperbolic_func(x):
    return 0.5 * torch.tanh(x) + 0.5


def arctan_func(x):
    return torch.arctan(x) / torch.pi + 0.5


def create_optimizer(args, net):
    optim_param = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if args.optimizer == "sgd":
        optim_param.update({"momentum": args.momentum, "nesterov": False})
        optimizer = torch.optim.SGD(net.parameters(), **optim_param)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(net.parameters(), **optim_param)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), **optim_param)
    else:
        raise NotImplementedError(f"{args.optimizer} is currently not implemented!")

    if args.use_casam:
        optimizer = SAM(
            net.parameters(), optimizer, rho=args.sam_rho, adaptive=args.adaptive
        )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.decay_gamma
        )
    elif args.scheduler == "const":
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=args.epochs
        )
        print("use constant learning rate, be careful!")
    else:
        raise NotImplementedError(f"{args.scheduler} is currently not implemented!")
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Consistency-Anchored SAM Optimizer")

    # data configuration
    parser.add_argument("--data_dir", default="/mnt/data1/public_dataset", type=str)
    parser.add_argument(
        "--data_name",
        default="cifar100",
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument("--num_classes", default=100, type=int)
    parser.add_argument("--corrupt_prob", default=0.0, type=float)
    parser.add_argument("--batch_size_train", default=128, type=int)
    parser.add_argument("--batch_size_eval", default=256, type=int)
    parser.add_argument("--use_cutout", default=False, type=bool)
    parser.add_argument(
        "--length", default=8, type=int
    )  # 16 for cifar-10, 8 for cifar-100
    parser.add_argument("--use_auto_augment", default=False, type=bool)
    parser.add_argument("--use_rand_augment", default=False, type=bool)
    parser.add_argument("--use_random_erasing", default=False, type=bool)
    parser.add_argument("--ls_alpha", default=0.0, type=float)
    parser.add_argument("--use_cutmix", default=0, type=int)
    parser.add_argument("--cutmix_alpha", default=1.0, type=float)
    parser.add_argument("--use_mixup", default=0, type=int)
    parser.add_argument("--mixup_alpha", default=0.2, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_memory", default=True, type=bool)

    # optimizer configuration
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5.0e-4, type=float)
    parser.add_argument(
        "--scheduler", default="cosine", choices=["cosine", "multistep", "const"]
    )
    parser.add_argument("--milestones", default=[100, 150], type=int, nargs="+")
    parser.add_argument("--decay_gamma", default=0.1, type=float)

    # architecture configuration
    parser.add_argument("--arch", default="resnet56_cifar100")
    # some other choices

    # train configuration
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--log_interval", default=1, type=int)
    parser.add_argument("--save_dir", default="runs", type=str)
    parser.add_argument("--restart", default=False, type=bool)
    parser.add_argument("--wandb", default=1, type=int)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--project", default="", type=str)

    parser.add_argument("--use_casam", default=0, type=int)
    parser.add_argument("--sam_rho", default=0.05, type=float)
    parser.add_argument("--casam_init_gamma", default=0.0, type=float)
    parser.add_argument("--casam_end_gamma", default=0.0, type=float)
    parser.add_argument("--adaptive", default=0, type=int)
    parser.add_argument("--best_conf", default=0, type=int)
    parser.add_argument(
        "--squeeze",
        default="none",
        type=str,
        choices=["none", "sigmoid", "arctan", "hyperbolic", "norm"],
    )

    args = parser.parse_args()

    if args.best_conf:
        args = select_best_conf(args)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    args.project = (
        f"{args.arch}_{args.optimizer}_{args.scheduler}_corruption_{args.corrupt_prob}_"
    )
    if args.use_cutout:
        args.project += "cutout"
    elif args.use_auto_augment:
        args.project += "auto_augment"
    elif args.use_rand_augment:
        args.project += "rand_augment"
    elif args.use_cutmix:
        args.project += "cutmix"
    elif args.use_mixup:
        args.project += "mixup"
    else:
        args.project += "basic"

    args.instance = f"corruption_{args.corrupt_prob}_ls_{args.ls_alpha}_"
    if args.use_casam:
        args.instance += f"casam_ada_{args.adaptive}_act={args.squeeze}gamma={args.casam_init_gamma}_{args.casam_end_gamma}_rho={args.sam_rho}_"
        gamma_sched = GammaScheduler(
            decay_steps=args.epochs,
            end_gamma=args.casam_end_gamma,
            initial_gamma=args.casam_init_gamma,
        )

    args.instance += f"lr={args.learning_rate:.2e}_bs={args.batch_size_train}_wd={args.weight_decay:.2e}_seed={args.seed}"

    # timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # save_dir = join(args.save_dir, args.project, args.instance, timestamp)
    save_dir = join(args.save_dir, args.project, args.arch, args.instance)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(join(save_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.wandb:
        # wandb_run = wandb.init(config=args, project=args.project, name=args.instance, settings=wandb.Settings(code_dir="."))
        wandb_run = wandb.init(
            config=args,
            project="casam2026",
            name=args.instance,
        )
        # wandb_run.log_code(".")

    init_random_state(args.seed)

    train_loader, test_loader, val_loader = get_cifar_dataloader(args)

    net = create_net(args).to(device)
    optimizer, scheduler = create_optimizer(args, net)

    if args.use_mixup or args.use_cutmix:
        train_loss_func = SoftCrossEntropyLoss(reduction="none")
    else:
        train_loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

    test_loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

    if args.use_casam:
        batch_updater = casam_batch_updater
    else:
        batch_updater = basic_batch_updater

    start_epoch = 0
    best_acc = 0.0
    logs = []

    with tqdm(total=args.epochs, colour="MAGENTA", ascii=True) as pbar:
        for epoch in range(start_epoch, args.epochs):
            if args.use_casam:
                args.casam_gamma = gamma_sched.get_gamma(epoch)
            train_loss, train_acc = trainer(
                net,
                optimizer,
                batch_updater,
                train_loader,
                train_loss_func,
                device,
                args,
            )
            scheduler.step()
            if epoch % args.log_interval == 0:

                val_loss, val_acc, val_logits, val_targets = validate(
                    net, val_loader, test_loss_func, device
                )

                logs.append([train_acc, val_acc, train_loss, val_loss])

                if best_acc <= val_acc:
                    best_acc = val_acc
                    best_net = deepcopy(net)
                # checkpoint = {
                #     "epoch": epoch,
                #     "model": net.state_dict(),
                #     "optimizer": optimizer.state_dict(),
                #     "lr_sched": scheduler,
                #     "logs": logs,
                #     "best_acc": best_acc,
                # }
                # if not os.path.exists(join(save_dir, "checkpoints")):
                #     os.makedirs(join(save_dir, "checkpoints"))
                # torch.save(
                #     net.state_dict(),
                #     join(save_dir, "checkpoints", f"checkpoint-{epoch}.pth.tar"),
                # )

                if args.wandb:
                    log_info = {
                        "train_acc": train_acc * 100,
                        "val_acc": val_acc * 100,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_acc": best_acc * 100,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    wandb_run.log(log_info)

                message = f"epoch: {epoch} "
                message += f"lr: {scheduler.get_last_lr()[0]:.6f} "
                message += f"train_loss: {Fore.RED}{train_loss:.4f}{Style.RESET_ALL} "
                message += (
                    f"train_acc: {Fore.RED}{train_acc*100:.2f}%{Style.RESET_ALL} "
                )
                message += f"val_loss: {Fore.GREEN}{val_loss:.4f}{Style.RESET_ALL} "
                message += f"test_acc: {Fore.GREEN}{val_acc*100:.2f}%{Style.RESET_ALL} "
                message += (
                    f"best_acc: {Fore.MAGENTA}{best_acc*100:.2f}%{Style.RESET_ALL} "
                )

                pbar.set_description(message)
                pbar.update()
    # logs = torch.tensor(logs)
    # torch.save(logs, join(save_dir, 'logs.pt'))
    torch.save(best_net.state_dict(), join(save_dir, "net_dict.pt"))
    # evluate test accuracy on the best validation set at the end of training
    best_net = best_net.to(device)
    val_loss, val_acc, val_logits, val_targets = validate(
        best_net, val_loader, test_loss_func, device
    )

    test_loss, test_acc, test_logits, test_targets = validate(
        best_net, test_loader, test_loss_func, device
    )

    torch.save(
        [test_loss, test_acc * 100, val_loss, val_acc * 100],
        join(save_dir, "metrics_on_test_set.pt"),
    )
    torch.save(
        [val_logits, val_targets, test_logits, test_targets], join(save_dir, "logs.pt")
    )
    print(
        f"metrics on val set: val loss ({val_loss:.4f}), val accuracy ({val_acc*100:.2f}%)."
    )
    print(
        f"metrics on test set: test loss ({test_loss:.4f}), test accuracy ({test_acc*100:.2f}%)."
    )
    if args.wandb:
        wandb_run.log({"test_acc": test_acc * 100, "test_loss": test_loss})


def trainer(net, optimizer, batch_updater, train_loader, train_loss_func, device, args):
    net.train()
    tot_loss = 0.0
    tot_correct = 0.0

    for i, batch in BackgroundGenerator(enumerate(train_loader)):
        loss, correct = batch_updater(
            net, optimizer, batch, train_loss_func, device, args
        )
        tot_loss += loss / len(train_loader.dataset)
        tot_correct += correct / len(train_loader.dataset)
    return tot_loss, tot_correct


def validate(net, test_loader, test_loss_func, device):
    net.eval()
    tot_loss = 0.0
    tot_correct = 0.0
    logits, targets = [], []
    num_examples = 0
    with torch.no_grad():
        for data, target in BackgroundGenerator(test_loader):
            num_examples += len(data)
            data, target = data.to(device), target.to(device)
            output = net(data)
            tot_loss += test_loss_func(output, target) * len(output)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            tot_correct += pred.eq(target.view_as(pred)).sum().item()
            logits.append(output)
            targets.append(target)
    targets = torch.hstack(targets).cpu()
    logits = torch.vstack(logits).cpu()
    return tot_loss / num_examples, tot_correct / num_examples, logits, targets


def basic_batch_updater(net, optimizer, batch, train_loss_func, device, args):
    optimizer.zero_grad()
    data, target = batch
    if not isinstance(target, (tuple, list)):
        target = target.to(device, non_blocking=True)
        data = data.to(device, non_blocking=True)
    output = net(data)
    loss = train_loss_func(output, target)
    loss.backward()
    optimizer.step()
    _, preds = torch.max(output.data, 1)
    if isinstance(target, (tuple, list)):
        targets1, targets2, lam = target
        targets1, targets2 = targets1.to(device), targets2.to(device)
        correct1 = preds.eq(targets1).sum().item()
        correct2 = preds.eq(targets2).sum().item()
        correct = lam * correct1 + (1 - lam) * correct2
    else:
        correct = preds.eq(target).sum().item()

    return loss.item() * len(data), correct


def casam_batch_updater(net, optimizer, batch, train_loss_func, device, args):
    optimizer.zero_grad()
    data, target = batch
    if not isinstance(target, (tuple, list)):
        target = target.to(device, non_blocking=True)
        data = data.to(device, non_blocking=True)

    # acent step
    logits = net(data)
    _, preds = torch.max(logits.data, 1)
    loss_prev = F.cross_entropy(
        logits, target, label_smoothing=args.ls_alpha, reduction="none"
    )

    loss = loss_prev.mean()
    loss.backward()

    optimizer.perturb_weight(zero_grad=True)

    # descent step
    logits = net(data)
    loss_post = F.cross_entropy(
        logits, target, label_smoothing=args.ls_alpha, reduction="none"
    )

    loss_prev_probs = (-loss_prev.data).exp()
    loss_post_probs = (-loss_post.data).exp()
    kl_div = bern_kl_div(loss_post_probs, loss_prev_probs)
    gap = loss_post.data - loss_prev.data - kl_div

    if args.squeeze == "sigmoid":
        gap = torch.sigmoid(gap)
    elif args.squeeze == "arctan":
        gap = arctan_func(gap)
    elif args.squeeze == "norm":
        gap = normalize_func(gap)
    elif args.squeeze == "hyperbolic":
        gap = hyperbolic_func(gap)
    else:
        raise NotImplementedError
    loss_post = (1.0 + gap) ** args.casam_gamma * loss_post

    loss_post = loss_post.mean()
    loss_post.backward()
    optimizer.step()

    if isinstance(target, (tuple, list)):
        targets1, targets2, lam = target
        targets1, targets2 = targets1.to(device), targets2.to(device)
        correct1 = preds.eq(targets1).sum().item()
        correct2 = preds.eq(targets2).sum().item()
        correct = lam * correct1 + (1 - lam) * correct2
    else:
        correct = preds.eq(target).sum().item()

    return loss.item() * len(data), correct


if __name__ == "__main__":
    main()
