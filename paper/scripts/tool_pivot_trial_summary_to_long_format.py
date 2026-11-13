import pandas as pd
import pathlib
from analyze_event_time_rates_trial import is_good_trial


def main(path: pathlib.Path):
    df = pd.read_csv(path, index_col=0)
    print(df.columns)
    print(f"Total number of trials: {len(df)}")

    drop_indices = []
    for i in range(len(df)):
        row = df.iloc[i]
        dataset_path = path.parent / row["recording"]
        if not is_good_trial(dataset_path, row, skip_ag_events=True):
            drop_indices.append(i)

    df = df.drop(drop_indices)
    df = df.reset_index(drop=True)

    print(f"Total number of good trials: {len(df)}")

    # Recode all columns that end with "_lose" and "_win" to start with "lose_" and "win_"
    for col in df.columns:
        if col.endswith("_lose"):
            prefix = str(col).split("_")[0]
            newname = f"lose_{prefix}"
            df.rename(columns={col: newname}, inplace=True)
        elif col.endswith("_win"):
            prefix = str(col).split("_")[0]
            newname = f"win_{prefix}"
            df.rename(columns={col: newname}, inplace=True)

    # Collect winner, loser and other columns
    win_cols = []
    lose_cols = []
    other_cols = []
    for col in df.columns:
        if col.startswith("lose_"):
            lose_cols.append(col)
        elif col.startswith("win_"):
            win_cols.append(col)
        else:
            other_cols.append(col)

    # Create subdf for winner including the other columns
    subdf_win = df[win_cols + other_cols]
    subdf_win = subdf_win.reset_index()
    subdf_win["status"] = 1

    # Create subdf for loser including the other columns
    subdf_lose = df[lose_cols + other_cols]
    subdf_lose = subdf_lose.reset_index()
    subdf_lose["status"] = 0

    # Remove "lose_" and "win_" in both dataframe col names
    subdf_win = subdf_win.rename(columns=lambda x: x.replace("win_", ""))
    subdf_lose = subdf_lose.rename(columns=lambda x: x.replace("lose_", ""))

    # add "paired_with_id" column before stacking
    winner_paired_with = subdf_lose["ID"]
    subdf_win["paired_fish"] = winner_paired_with
    loser_paired_with = subdf_win["ID"]
    subdf_lose["paired_fish"] = loser_paired_with

    # rename "ID"
    subdf_win = subdf_win.rename(columns={"ID": "focal_fish"})
    subdf_lose = subdf_lose.rename(columns={"ID": "focal_fish"})
    print(subdf_win)

    print(subdf_win)
    print(subdf_lose)

    # Pivot to long format
    df = pd.concat([subdf_win, subdf_lose])
    df = df.rename(columns={"index": "trial_idx"})

    print("Dataframe columns: ---------")
    print(df.columns)

    targetpath = pathlib.Path("data/processed/trial_summary_long.csv")
    # df.to_csv(targetpath)


if __name__ == "__main__":
    path = pathlib.Path("/mnt/data2/2019_tube_competition/trial_summary.csv")
    main(path)
