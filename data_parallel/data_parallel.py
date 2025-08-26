import torch
import torch.nn as nn
import torchvision
from torchvision import models, datasets
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset

import numpy as np

import time
from tqdm import tqdm

import config as cfg

import random


def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: If you use cudnn backend and want to be even more rigorous
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed}")


def reset_peaks_all(devices=None):
    """Reset peak memory stats on all (visible) CUDA devices."""
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    for d in devices:
        torch.cuda.reset_peak_memory_stats(d)


def peek_peaks_all(devices=None):
    """Read peak allocated / reserved (in GiB) for all devices."""
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    per_dev = []
    for d in devices:
        alloc_gib = torch.cuda.max_memory_allocated(d) / (1024**3)
        reserv_gib = torch.cuda.max_memory_reserved(d) / (1024**3)
        per_dev.append((d, round(alloc_gib, 3), round(reserv_gib, 3)))
    return per_dev


def print_peaks(per_dev, prefix=""):
    print(f"{prefix}Peak memory (GiB) per device [allocated / reserved]:")
    for d, alloc, reserv in per_dev:
        print(f"{prefix}  cuda:{d}: {alloc:.3f} / {reserv:.3f}")
    if per_dev:
        max_alloc = max(x[1] for x in per_dev)
        max_res = max(x[2] for x in per_dev)
        sum_alloc = sum(x[1] for x in per_dev)
        sum_res = sum(x[2] for x in per_dev)
        print(
            f"{prefix}  (max across devices)  alloc={max_alloc:.3f}, reserved={max_res:.3f}"
        )
        print(
            f"{prefix}  (sum across devices)  alloc={sum_alloc:.3f}, reserved={sum_res:.3f}"
        )


def warmup_training(
    model,
    loader,
    optimizer,
    criterion,
    device,
    num_warmup_batches=0,
    warmup_full_epoch=False,
):
    """
    Run a brief warmup so cuDNN autotune / kernels / workers stabilize.
    Does real forward/backward/step, just not timed/logged.
    """
    if num_warmup_batches <= 0 and not warmup_full_epoch:
        return

    model.train()
    it = iter(loader)
    if warmup_full_epoch:
        # consume a full epoch
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    else:
        # consume a few batches
        for _ in range(min(num_warmup_batches, len(loader))):
            x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


def time_train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    device_ids,
    do_warmup=False,
    warmup_batches=0,
    warmup_full_epoch=False,
):
    """
    Times a single *training* epoch with proper CUDA sync + peak memory capture.
    Optionally performs a warmup right before timing.
    """
    if do_warmup:
        warmup_training(
            model,
            loader,
            optimizer,
            criterion,
            device,
            num_warmup_batches=warmup_batches,
            warmup_full_epoch=warmup_full_epoch,
        )

    model.train()
    for d in device_ids:
        torch.cuda.synchronize(d)
    reset_peaks_all(device_ids)

    t0 = time.perf_counter()
    loss_sum = 0.0

    for x, y in tqdm(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    for d in device_ids:
        torch.cuda.synchronize(d)
    elapsed = round(time.perf_counter() - t0, 3)

    per_device_peaks = peek_peaks_all(device_ids)

    return {
        "time_s": round(elapsed, 3),
        "loss": loss_sum / len(loader),
        "per_device_peaks": per_device_peaks,
    }


@torch.no_grad()
def time_test_epoch(
    model,
    loader,
    criterion,
    device,
    device_ids,
):
    """
    Times a single *test* epoch (no grad) with proper CUDA sync + peak memory capture.
    """
    model.eval()
    for d in device_ids:
        torch.cuda.synchronize(d)
    reset_peaks_all(device_ids)

    t0 = time.perf_counter()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in tqdm(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    for d in device_ids:
        torch.cuda.synchronize(d)
    elapsed = time.perf_counter() - t0

    per_device_peaks = peek_peaks_all(device_ids)

    return {
        "time_s": round(elapsed, 3),
        "loss": loss_sum / len(loader),
        "accuracy": correct / total,
        "per_device_peaks": per_device_peaks,
    }


def data_parallel_main(args):
    do_data_parallel = args["do_data_parallel"]

    batch_size = args["batch_size"]
    dataloader_num_workers = args["dataloader_num_workers"]
    train_data_size = args["train_data_size"]
    test_data_size = args["test_data_size"]
    learning_rate = args["learning_rate"]
    epochs = args["epochs"]
    device = torch.device(args["device"])
    visible_devices = cfg.visible_devices
    imagenette_train_path = args["imagenette_train_path"]
    imagenette_test_path = args["imagenette_test_path"]

    torch.cuda.set_per_process_memory_fraction(cfg.memory_limit)

    test_transform = models.ResNet152_Weights.DEFAULT.transforms()
    train_transform = T.Compose(
        [
            T.ToImageTensor(),
            T.RandomResizedCrop(224, antialias=True),
            T.RandomHorizontalFlip(),
            T.ConvertDtype(torch.float32),
            T.Normalize(test_transform.mean, test_transform.std),
        ]
    )

    train_dataset = datasets.ImageFolder(
        imagenette_train_path, transform=train_transform
    )
    test_dataset = datasets.ImageFolder(imagenette_test_path, transform=test_transform)

    if train_data_size < len(train_dataset):
        print("Using subset for train_dataset")
        train_dataset = Subset(train_dataset, range(train_data_size))

    if test_data_size < len(test_dataset):
        print("Using subset for test_dataset")
        test_dataset = Subset(test_dataset, range(test_data_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )

    # # if we want to randomly select instead
    # random_train_indices = np.random.choice(
    #     len(train_dataset), train_data_size, replace=False
    # )
    # random_train_sampler = SubsetRandomSampler(random_train_indices)

    # random_test_indices = np.random.choice(
    #     len(test_dataset), test_data_size, replace=False
    # )
    # random_test_sampler = SubsetRandomSampler(random_test_indices)

    # random_train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     sampler=random_train_sampler,
    #     num_workers=dataloader_num_workers,
    #     pin_memory=True,
    # )
    # random_test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     sampler=random_test_sampler,
    #     num_workers=dataloader_num_workers,
    #     pin_memory=True,
    # )

    # print(f"{random_train_indices = }")
    # print(f"{random_test_indices = }")

    # # if we want to run a quick sanity check with the first 4 images
    # sanity_indices = list(range(4))
    # sanity_dataset = Subset(test_dataset, sanity_indices)

    # # Create a DataLoader for the sanity dataset
    # sanity_loader = DataLoader(
    #     sanity_dataset,
    #     batch_size=cfg.per_device_batch_size,
    #     shuffle=False,  # No need to shuffle for a fixed sanity set
    #     num_workers=cfg.dataloader_num_workers,
    #     pin_memory=True,
    # )

    # print(f"Created sanity_dataset with {len(sanity_dataset)} images.")

    print("Loading model...")

    model = models.resnet152(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    if do_data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=visible_devices)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("Training on " + str(device))
    t0 = time.perf_counter()

    for epoch in tqdm(range(epochs)):
        train_stats = time_train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            visible_devices,
            do_warmup=(epoch == 1),
            warmup_batches=2,
        )
        test_stats = time_test_epoch(
            model, test_loader, criterion, device, visible_devices
        )

        scheduler.step()
        print(
            "\n"
            f"Epoch {epoch + 1}/{epochs} | "
            f"train: loss {train_stats['loss']:.4f}, time {train_stats['time_s']:.1f}s | "
            f"test: loss {test_stats['loss']:.4f}, acc {test_stats['accuracy']:.3f}, time {test_stats['time_s']:.1f}s"
        )
        print("• train:")
        print_peaks(train_stats["per_device_peaks"], prefix="  ")
        print("• test:")
        print_peaks(test_stats["per_device_peaks"], prefix="  ")

    print(
        f"Time taken per epoch (seconds) {((time.perf_counter() - t0) / epochs):.2f}s"
    )

    return {
        "loss": train_stats["loss"],
        "per_device_peaks": {
            "train": train_stats["per_device_peaks"],
            "test": test_stats["per_device_peaks"],
        },
    }


if __name__ == "__main__":
    total_devices = len(cfg.visible_devices) if cfg.do_data_parallel else 1
    print(f"Training on {total_devices} devices")
    batch_size = cfg.per_device_batch_size * total_devices
    print("Per Device Batch Size = ", cfg.per_device_batch_size)
    print("Total Effective Batch Size =", batch_size)

    dataloader_num_workers = cfg.dataloader_num_workers
    print("Number of workers for dataloaders = ", dataloader_num_workers)

    args = {
        "do_data_parallel": cfg.do_data_parallel,
        "batch_size": batch_size,
        "dataloader_num_workers": dataloader_num_workers,
        "learning_rate": cfg.learning_rate,
        "epochs": cfg.epochs,
        "device": cfg.device,
        "imagenette_train_path": cfg.imagenette_train_path,
        "imagenette_test_path": cfg.imagenette_test_path,
        "train_data_size": cfg.train_data_size,
        "test_data_size": cfg.test_data_size,
    }

    tv_model_path = cfg.tv_model_path
    torch.hub.set_dir(tv_model_path)

    set_seed(42)
    torch.backends.cudnn.benchmark = True

    print(f"{torch.__version__ = }")
    print(f"{torchvision.__version__ = }")
    print(f"{torch.cuda.is_available() = }")
    print(f"{torch.cuda.device_count() = }")
    print(f"{torch.cuda.get_device_name(0) = }")

    results = data_parallel_main(args)
    print(f"Final loss = {results['loss']}")

    train_peak_memory = results["per_device_peaks"]["train"]
    test_peak_memory = results["per_device_peaks"]["test"]
    max_memory_consumed = max(
        max(x[1] for x in train_peak_memory), max(x[1] for x in test_peak_memory)
    )
    max_memory_consumed = round(max_memory_consumed * 1.073741824, 2)
    print(f"Max Memory Consumed Per Device = {max_memory_consumed} GB")
