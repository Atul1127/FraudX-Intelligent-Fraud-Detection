# Data Directory

The IEEE-CIS Fraud Detection dataset is intentionally not committed to Git because of its size and Kaggle distribution restrictions.

Place the downloaded files in `data/raw/`:

```text
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

Then run:

```bash
python train.py
```

The training pipeline validates required files and reports exactly which files are missing if the dataset has not been installed locally.
