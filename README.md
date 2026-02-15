# PC-Mix

PC-Mix is a partial-component spoofing dataset under mixed speech–background conditions, with a baseline model built upon BAM.

---

## Installation

Create environment and install dependencies:

```bash
conda create -n pcmix python=3.8
conda activate pcmix
pip install -r requirements.txt
```


---

## Usage

### Data Preparation

Modify dataset path in:

```
dataset/ps_preprocess.py
```

Run:

```bash
python dataset/ps_preprocess.py
```

---

### Training

```bash
python train.py --train_root ./data/raw/train --dev_root ./data/raw/dev
```

---

### Evaluation

```bash
python train.py --test_only --checkpoint ./bam_checkpoint/model.ckpt --eval_root ./data/raw/eval
```

---

## Dataset

Coming Soon.

---

## Paper

Coming Soon.
