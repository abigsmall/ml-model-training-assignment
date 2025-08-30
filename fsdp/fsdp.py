import torch
import torch.nn as nn
import torchvision
from torchvision import models, datasets
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

import numpy as np

import time
from tqdm import tqdm

import config as cfg

import random
import os
import functools


# Initialize the distributed process group
def setup():
    dist.init_process_group("nccl")

# Cleanup the distributed process group
def cleanup():
    dist.destroy_process_group()

def set_seed(seed, rank):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: If you use cudnn backend and want to be even more rigorous
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    if (rank == 0):
        print(f"Seeds set to {seed}")


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


def gather_peak_memory_gib(local_device, group=None, return_on_all_ranks=False):
    # local peaks (GiB)
    alloc_gib = torch.cuda.max_memory_allocated(local_device) / (1024**3)
    reserv_gib = torch.cuda.max_memory_reserved(local_device) / (1024**3)
    local = torch.tensor([alloc_gib, reserv_gib], device=local_device)

    if not dist.is_available() or not dist.is_initialized():
        return np.array([[alloc_gib, reserv_gib]])

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local, group=group)

    arr = torch.stack(gathered).cpu().numpy()
    if return_on_all_ranks or rank == 0:
        return arr
    return None


def peaks_from_gather(arr):
    # arr: np.ndarray [world_size, 2]
    return [(i, float(a), float(r)) for i, (a, r) in enumerate(arr)]


def warmup_training(
    model,
    rank,
    loader,
    optimizer,
    criterion,
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
            x, y = x.to(rank, non_blocking=True), y.to(rank, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    else:
        # consume a few batches
        for _ in range(min(num_warmup_batches, len(loader))):
            x, y = next(it)
            x, y = x.to(rank, non_blocking=True), y.to(rank, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


def time_train_epoch(
    model,
    rank,
    loader,
    optimizer,
    criterion,
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
            rank,
            loader,
            optimizer,
            criterion,
            num_warmup_batches=warmup_batches,
            warmup_full_epoch=warmup_full_epoch,
        )

    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    # 0: loss_sum, 1: total
    loss_sum = torch.zeros(3).to(rank)

    for batch in tqdm(loader):
        x, y = batch
        x, y = x.to(rank, non_blocking=True), y.to(rank, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum[0] += loss.item()
        loss_sum[1] += len(batch)
        loss_sum[2] += y.size(0)

    
    torch.cuda.synchronize()
    elapsed = round(time.perf_counter() - t0, 3)

    # Sum the loss across all distributed processes
    print(f"train: before reduce: rank: {rank} | len(loader.dataset): {len(loader.dataset)} | len(loader): {len(loader)} | loss_sum[0]: {loss_sum[0]} | loss_sum[1]: {loss_sum[1]} | loss_sum[2]: {loss_sum[2]}")
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    print(f"train: after reduce: rank: {rank} | len(loader.dataset): {len(loader.dataset)} | len(loader): {len(loader)} | loss_sum[0]: {loss_sum[0]} | loss_sum[1]: {loss_sum[1]} | loss_sum[2]: {loss_sum[2]}")

    arr = gather_peak_memory_gib(rank, return_on_all_ranks=False)
    per_device_peaks = []
    if arr is not None:
        per_device_peaks = peaks_from_gather(arr)

    return {
        "time_s": round(elapsed, 3),
        "loss": loss_sum[0] / len(loader),
        "per_device_peaks": per_device_peaks,
    }


@torch.no_grad()
def time_test_epoch(
    model,
    rank,
    loader,
    criterion,
):
    """
    Times a single *test* epoch (no grad) with proper CUDA sync + peak memory capture.
    """
    model.eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    # 0: total, 1: correct, 2: loss_sum
    eval_stats = torch.zeros(3).to(rank)

    for x, y in tqdm(loader):
        x, y = x.to(rank, non_blocking=True), y.to(rank, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        eval_stats[0] += y.size(0)
        eval_stats[1] += (logits.argmax(1) == y).sum().item()
        eval_stats[2] += loss.item()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # debug:
    print(f"rank: {rank} | total: {eval_stats[0]} | len(loader.dataset): {len(loader.dataset)} | len(loader): {len(loader)}")
    dist.all_reduce(eval_stats, op=dist.ReduceOp.SUM)
    print(f"rank: {rank} | reduced total: {eval_stats[0]} | reduced correct: {eval_stats[1]} | reduced loss_sum: {eval_stats[2]}")

    arr = gather_peak_memory_gib(rank, return_on_all_ranks=False)
    per_device_peaks = []
    if arr is not None:
        per_device_peaks = peaks_from_gather(arr)

    return {
        "time_s": round(elapsed, 3),
        "loss": eval_stats[2] / len(loader),
        "accuracy": eval_stats[1] / eval_stats[0],
        "per_device_peaks": per_device_peaks,
    }


def fsdp_main(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    batch_size = args["per_device_batch_size"]
    dataloader_num_workers = args["dataloader_num_workers"]
    train_data_size = args["train_data_size"]
    test_data_size = args["test_data_size"]
    learning_rate = args["learning_rate"]
    epochs = args["epochs"]
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

    if train_data_size != -1 and train_data_size < len(train_dataset):
        if local_rank == 0:
            print("Using subset for train_dataset")
        train_dataset = Subset(train_dataset, range(train_data_size))

    if test_data_size != -1 and test_data_size < len(test_dataset):
        if local_rank == 0:
            print("Using subset for test_dataset")
        test_dataset = Subset(test_dataset, range(test_data_size))

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)

    setup()

    train_kwargs = {"batch_size": batch_size, "num_workers": dataloader_num_workers, "sampler": train_sampler, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 4}
    test_kwargs = {"batch_size": batch_size, "num_workers": dataloader_num_workers, "sampler": test_sampler, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 4}

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    # # if we want to run a quick sanity check with the first 4 images
    # sanity_indices = list(range(4))
    # sanity_dataset = Subset(test_dataset, sanity_indices)

    # # Create a DataLoader for the sanity dataset
    # sanity_loader = DataLoader(
    #     sanity_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,  # No need to shuffle for a fixed sanity set
    #     num_workers=cfg.dataloader_num_workers,
    #     pin_memory=True,
    # )

    # if local_rank == 0:
    #     print(f"Created sanity_dataset with {len(sanity_dataset)} images.")

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )

    if local_rank == 0:
        print("Loading model...")

    torch.cuda.set_device(local_rank)

    model = models.resnet152(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 10)

    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=torch.cuda.current_device())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_stats = time_train_epoch(
            model,
            rank,
            train_loader,
            optimizer,
            criterion,
            do_warmup=(epoch == 1),
            warmup_batches=2,
        )
        test_stats = time_test_epoch(
            model, rank, test_loader, criterion
        )

        scheduler.step()

        if local_rank == 0:
            print(
                "\n"
                f"Epoch {epoch}/{epochs} | "
                f"train: loss {train_stats['loss']:.4f}, time {train_stats['time_s']:.1f}s | "
                f"test: loss {test_stats['loss']:.4f}, acc {test_stats['accuracy']:.3f}, time {test_stats['time_s']:.1f}s"
            )
            print("• train:")
            print_peaks(train_stats["per_device_peaks"], prefix="  ")
            print("• test:")
            print_peaks(test_stats["per_device_peaks"], prefix="  ")

    # Synchronize all distributed processes
    dist.barrier()
    cleanup()

    elapsed = time.perf_counter() - t0
    time_per_epoch_s = elapsed / epochs
    if local_rank == 0:
        print(
            f"Time taken per epoch (seconds) {time_per_epoch_s:.2f}s"
        )

    return {
        "loss": train_stats["loss"],
        "time_per_epoch_s": time_per_epoch_s,
        "per_device_peaks": {
            "train": train_stats["per_device_peaks"],
            "test": test_stats["per_device_peaks"],
        },
    }


if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    total_devices = int(os.environ['WORLD_SIZE'])

    if local_rank == 0:
        print(f"Training on {total_devices} devices")
    batch_size = cfg.per_device_batch_size * total_devices

    if local_rank == 0:
        print("Per Device Batch Size = ", cfg.per_device_batch_size)
        print("Total Effective Batch Size =", batch_size)

    dataloader_num_workers = cfg.dataloader_num_workers
    if local_rank == 0:
        print("Number of workers for dataloaders = ", dataloader_num_workers)

    args = {
        "per_device_batch_size": cfg.per_device_batch_size,
        "dataloader_num_workers": dataloader_num_workers,
        "learning_rate": cfg.learning_rate,
        "epochs": cfg.epochs,
        "imagenette_train_path": cfg.imagenette_train_path,
        "imagenette_test_path": cfg.imagenette_test_path,
        "train_data_size": cfg.train_data_size,
        "test_data_size": cfg.test_data_size,
    }

    tv_model_path = cfg.tv_model_path
    torch.hub.set_dir(tv_model_path)

    set_seed(42, local_rank)
    torch.backends.cudnn.benchmark = True

    if local_rank == 0:
        print(f"{torch.__version__ = }")
        print(f"{torchvision.__version__ = }")
        print(f"{torch.cuda.is_available() = }")
        print(f"{torch.cuda.device_count() = }")
        print(f"{torch.cuda.get_device_name(0) = }")

    results = fsdp_main(args)

    if local_rank == 0:
        loss = results['loss']
        print(f"Final loss = {loss}")

        train_peak_memory = results["per_device_peaks"]["train"]
        test_peak_memory = results["per_device_peaks"]["test"]

        max_memory_consumed = max(
            max(x[1] for x in train_peak_memory), max(x[1] for x in test_peak_memory)
        )
        max_memory_consumed = round(max_memory_consumed * 1.073741824, 2)
        print(f"Max Memory Consumed Per Device = {max_memory_consumed} GB")
        print(f"loss: {loss:.4f} | {results['time_per_epoch_s']:.2f}s | {max_memory_consumed} GB")
