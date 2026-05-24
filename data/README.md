# Data

## Source

**Totally-Looks-Like (TLL)** dataset — collected from Reddit r/totallynotrobots, curated for the paper:

> *Totally Looks Like — How Humans Compare, Compared to Machines*  
> Rosenfeld et al., CVPR 2019

## Setup

Download the dataset and place it under `data/` with the following structure:

```
data/
├── train.csv                  # 2000 image pair labels (committed)
├── test_candidates.csv        # test query → 20 candidates mapping (committed)
├── sample-submission.csv      # submission format reference (committed)
├── train/
│   ├── left/                  # 2000 anchor images (.jpg)
│   └── right/                 # 2000 positive images (.jpg)
└── test/
    ├── left/                  # 2000 test query images
    └── right/                 # candidate pool
```

Image directories are excluded from git (`.gitignore`).  
CSV files defining splits and candidates are committed for reproducibility.
