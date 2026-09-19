"""
Loading the chest X-ray dataset.

Two things here are easy to get wrong and both change the reported numbers.

**Augmentation belongs to training only.** Random rotation on the test set is
test-time augmentation: it perturbs the images you are scoring on, so the
number you report is not the model's accuracy on the data, it is the model's
accuracy on a randomly jittered version of the data, and it changes between
runs. Only `train` gets random transforms here.

**The classes are not balanced.** The training split is roughly 3:1
pneumonia:normal. A model that answers "pneumonia" unconditionally scores
about 74 % accuracy while being clinically useless, which is why `train.py`
reports recall and precision per class rather than accuracy alone, and why
the loss is class-weighted.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T

# ImageNet channel statistics. Used because the normalization these images are
# scaled to should match whatever a pretrained backbone would expect, which
# keeps the door open to swapping in a pretrained model later.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RESIZE = 256
CROP = 224

SPLITS = ("train", "val", "test")


def transforms(split):
    """Transform pipeline for one split.

    Raises
    ------
    ValueError
        On an unknown split name, rather than silently returning the training
        pipeline and quietly augmenting the test set.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    common = [
        T.Resize((RESIZE, RESIZE)),
        T.CenterCrop(CROP),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    if split == "train":
        return T.Compose(
            [
                T.Resize((RESIZE, RESIZE)),
                T.RandomRotation(degrees=(-20, 20)),
                T.RandomHorizontalFlip(),
                T.CenterCrop(CROP),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return T.Compose(common)


def find_data_root(data_dir):
    """Locate the directory that actually holds train/ val/ test/.

    The Kaggle archive unzips to a doubled `chest_xray/chest_xray/` in some
    versions and a single `chest_xray/` in others, which is a reliable way to
    waste twenty minutes. This checks both.
    """
    root = Path(data_dir)
    candidates = [root, root / "chest_xray"]

    for candidate in candidates:
        if all((candidate / split).is_dir() for split in SPLITS):
            return candidate

    raise FileNotFoundError(
        f"Could not find train/ val/ test/ under {root} or {root / 'chest_xray'}.\n"
        f"Download the dataset first, see the README."
    )


def loaders(data_dir, batch_size=32, num_workers=2):
    """Build the three DataLoaders.

    Returns
    -------
    (dict of str -> DataLoader, list of str)
        The loaders and the class names, in the index order the model's
        output columns use.
    """
    root = find_data_root(data_dir)

    sets = {
        split: datasets.ImageFolder(root / split, transform=transforms(split))
        for split in SPLITS
    }

    out = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in sets.items()
    }

    return out, sets["train"].classes


def class_weights(loader, n_classes=2):
    """Inverse-frequency weights, for a class-weighted loss.

    Computed from the dataset's targets rather than by iterating the loader,
    because iterating would decode every image just to count labels.
    """
    targets = torch.tensor(loader.dataset.targets)
    counts = torch.bincount(targets, minlength=n_classes).float()
    weights = counts.sum() / (n_classes * counts.clamp(min=1))
    return weights
