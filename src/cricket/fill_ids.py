"""Replace PUT_ID placeholders with the Cricsheet ID of the squad player with the same name."""

import argparse

import pandas as pd

PLACEHOLDER = "PUT_ID"


def fill_ids(table, squads):
    """Return a copy of table with each PUT_ID filled from squads, plus a message for every row tried."""
    ids_by_name = {}
    for _, row in squads.iterrows():
        if pd.notna(row["cricsheet_id"]):
            ids_by_name.setdefault(row["player_name"], []).append(row["cricsheet_id"])
    filled = table.copy()
    messages = []
    for i, row in filled.iterrows():
        if row["cricsheet_id"] != PLACEHOLDER:
            continue
        name = row["player_name"]
        found = ids_by_name.get(name, [])
        if len(found) == 1:
            filled.at[i, "cricsheet_id"] = found[0]
            messages.append(f"filled   {name}: {found[0]}")
        elif not found:
            messages.append(f"NOT FILLED {name}: no squad player with exactly this name and an ID")
        else:
            messages.append(f"NOT FILLED {name}: more than one squad player has this name")
    return filled, messages


def fill_file(path, squads):
    """Fill the PUT_IDs in one CSV file, save it, and return the messages."""
    table = pd.read_csv(path, dtype={"cricsheet_id": str})
    filled, messages = fill_ids(table, squads)
    filled.to_csv(path, index=False)
    return messages


def main():
    parser = argparse.ArgumentParser(description="Fill PUT_ID placeholders from the squads file.")
    parser.add_argument("files", nargs="+", help="CSV files containing PUT_ID placeholders")
    parser.add_argument("--squads", default="data/manual/squads_2027.csv")
    args = parser.parse_args()
    squads = pd.read_csv(args.squads, dtype={"cricsheet_id": str})
    for path in args.files:
        print(f"--- {path}")
        for message in fill_file(path, squads):
            print(message)


if __name__ == "__main__":
    main()
