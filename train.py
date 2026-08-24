from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection ensemble")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Recompute features even if cache exists",
    )
    parser.add_argument(
        "--skip-smote",
        action="store_true",
        help="Skip SMOTE oversampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.data.loader import (
        load_raw,
        train_val_split,
        save_processed,
        load_processed,
        processed_exists,
    )
    from src.data.features import build_features, apply_smote, fit_category_mappings
    from src.train import Trainer

    proc = Path(cfg["data"]["processed_dir"])
    target = cfg["features"]["target_col"]
    category_mappings = {}

    if not args.force_preprocess and processed_exists(cfg):
        print("Loading cached features...")
        X_train, y_train = load_processed(proc / "features_train.pkl")
        X_val, y_val = load_processed(proc / "features_val.pkl")

        # The cached matrix is safe to reuse, but online inference still needs
        # the category vocabulary that produced the original numeric codes.
        print("Loading raw data to recover categorical mappings for serving...")
        raw_for_metadata = load_raw(cfg)
        category_mappings = fit_category_mappings(raw_for_metadata)
        print(f"  Recovered mappings for {len(category_mappings)} categorical columns.")
    else:
        print("Loading raw data...")
        df = load_raw(cfg)
        print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
        print(f"  Fraud rate: {df[target].mean():.4f}")

        print("Fitting categorical mappings...")
        category_mappings = fit_category_mappings(df)
        print(f"  Saved mappings for {len(category_mappings)} categorical columns.")

        print("Engineering features...")
        df_feat = build_features(df, cfg, category_mappings=category_mappings)
        print(f"  Feature matrix: {df_feat.shape}")

        X_train, X_val, y_train, y_val = train_val_split(df_feat, cfg)
        print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}")

        save_processed((X_train, y_train), proc / "features_train.pkl")
        save_processed((X_val, y_val), proc / "features_val.pkl")
        print("  Cached processed features.")

    if not args.skip_smote:
        print("Applying SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train, cfg)

    print("\nTraining ensemble...")
    trainer = Trainer(cfg)
    trainer.set_category_mappings(category_mappings)
    report = trainer.run(X_train, y_train, X_val, y_val)

    ckpt_dir = Path("models/checkpoints")
    trainer.save(ckpt_dir, report)

    # Persist best threshold back to config
    cfg["ensemble"]["default_threshold"] = report["best_threshold"]
    with open(args.config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\nBest threshold ({report['best_threshold']:.3f}) written to {args.config}")
    print("Done.")


if __name__ == "__main__":
    main()
