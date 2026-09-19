# Pneumonia detection from chest radiographs

A convolutional network that classifies paediatric chest X-rays as NORMAL or
PNEUMONIA, trained from scratch in PyTorch on the Kermany et al. dataset.

The interesting part of this problem is not the architecture. It is that the
dataset is roughly **3:1 pneumonia:normal**, which means a model that answers
"pneumonia" unconditionally scores about 74 % accuracy while being clinically
worthless, it never identifies a healthy patient. So this reports a full
confusion matrix and per-class precision/recall, and trains against a
class-weighted loss. Accuracy alone cannot tell you whether the model works.

## Quickstart

The dataset is **not** in this repository, see [Data](#data).

```bash
pip install -r requirements.txt
python train.py --data-dir path/to/chest_xray
```

```
Device        : cuda
Classes       : ['NORMAL', 'PNEUMONIA']
train         : 5216 images
val           : 16 images
test          : 624 images
Class weights : [1.94, 0.67]
epoch   1/15  train 0.4127  val 0.3319  val_acc 87.5 %
...

=== Test set ===
  Confusion matrix (rows = true, cols = predicted)
                    NORMAL   PNEUMONIA
  NORMAL               ...         ...
  PNEUMONIA            ...         ...

  class          precision      recall          f1     support
  NORMAL             ...         ...         ...         234
  PNEUMONIA          ...         ...         ...         390
```

The number that matters clinically is **recall on PNEUMONIA**: a false
negative is a missed diagnosis.

## Architecture

Five conv blocks, then a classifier head.

```
input 3 x 224 x 224
  conv 3 -> 32    BN  ReLU           pool -> 112
  conv 32 -> 64   BN  ReLU  drop 0.1 pool ->  56
  conv 64 -> 64   BN  ReLU           pool ->  28
  conv 64 -> 128  BN  ReLU  drop 0.2 pool ->  14
  conv 128 -> 256 BN  ReLU  drop 0.2 pool ->   7
  flatten                    256 x 7 x 7 = 12544
  linear 12544 -> 128  ReLU  drop 0.3
  linear 128 -> 2                          logits
```

2,032,450 trainable parameters.

The flatten size is computed in `__init__` from the input size and the number
of pools, not discovered during the first forward pass. Discovering it lazily
is tempting and it is a trap: the layer gets created *after* the optimizer has
already been handed `model.parameters()`, so its weights never enter an update
step and stay at their random initialization for the entire run.

## Layout

```
model.py    the network, and the flatten-size arithmetic
data.py     transforms, loaders, class weights
train.py    training loop, evaluation, the metrics report
```

## Data

Chest X-Ray Images (Pneumonia), 5,863 images across NORMAL and PNEUMONIA.

> Kermany, D.S., Goldbaum, M., et al. Identifying Medical Diagnoses and
> Treatable Diseases by Image-Based Deep Learning. *Cell* 172(5):1122-1131,
> 2018.

Download it from
[Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
or [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2) (CC BY 4.0)
and unzip it anywhere, then point `--data-dir` at it.

Some versions of the archive unzip to a doubled `chest_xray/chest_xray/` and
some to a single `chest_xray/`. `data.py` checks for both, so either works.

The images are not committed here. A 1.2 GB dataset in git history is a
permanent download for everyone who clones the repo, and Kaggle and Mendeley
are stable canonical sources.

## Caveats

- **`val` has 16 images.** That is the split as the dataset ships, and it is
  far too small to select a checkpoint on reliably. A proper run would re-split
  train into train/val and keep the official test set untouched.
- **Trained from scratch.** A pretrained ResNet or DenseNet backbone
  fine-tuned on this data outperforms this network substantially, and for
  5,000 images that is the sensible choice. Training from scratch here is a
  deliberate exercise, not a claim to be the best approach.
- **Paediatric data only.** All images are from patients aged 1-5 at one
  medical centre in Guangzhou. Nothing here should be expected to transfer to
  adult radiographs or to another site's imaging protocol.
- **Not a medical device.** Obviously, but worth stating: this is a learning
  exercise on a public dataset, not a diagnostic tool.


## What changed from the first version

The original commit had a single script with a model whose `forward()` built
a fully-connected layer and then returned the flattened conv features without
ever applying it, so the network emitted 12,544-dimensional vectors that
`CrossEntropyLoss` scored as logits over 12,544 classes, and any accuracy it
printed was meaningless. It also hardcoded an absolute path to a local
directory, applied random rotation to the test set, and reported accuracy on a
3:1 imbalanced problem with no other metric.

This version fixes those, splits the script into three modules, and drops the
dataset out of git history.

## License

MIT, see [LICENSE](LICENSE). The dataset is not covered by it; it is
distributed under CC BY 4.0 by its authors.
