from data.split import build_splits

LOW_DRIFT_PFL_PATH = "data/processed/PFL1_low_drift_devices.csv"
PFL1_PATH          = "data/processed/PFL1_preprocessed.csv"


if __name__ == "__main__":
    print("Starting pipeline...")
    pfl1_df, split_map = build_splits(LOW_DRIFT_PFL_PATH, PFL1_PATH)
    # next: train encoder