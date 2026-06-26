"""
Common functions for processors.
"""

from __future__ import annotations

# In src/hbb/utils.py
import pickle
import warnings
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
from coffea.analysis_tools import PackedSelection

P4 = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}

def add_selection(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
    cutflow: dict,
    isData: bool,
    genWeights: ak.Array = None,
):
    """adds selection to PackedSelection object and the cutflow dictionary"""
    if isinstance(sel, ak.Array):
        sel = sel.to_numpy()

    selection.add(name, sel.astype(bool))
    cutflow[name] = (
        np.sum(selection.all(*selection.names))
        if isData
        # add up genWeights for MC
        else np.sum(genWeights[selection.all(*selection.names)])
    )


def check_selector(sample: str, selector: str | list[str]):
    if not isinstance(selector, (list, tuple)):
        selector = [selector]

    for s in selector:
        if s.endswith("?"):
            if s[:-1] == sample:
                return True
        elif s.startswith("*"):
            if s[1:] in sample:
                return True
        else:
            if sample.startswith(s):
                return True

    return False


def get_sum_genweights(data_dir: Path, dataset: str) -> float:
    """
    Get the sum of genweights for a given dataset.
    :param data_dir: The directory where the datasets are stored.
    :param dataset: The name of the dataset to get the genweights for.
    :return: The sum of genweights for the dataset.
    """
    total_sumw = 0

    try:
        # Load the genweights from the pickle file
        for pickle_file in list(Path(data_dir / dataset / "pickles").glob("*.pkl")):
            with Path(pickle_file).open("rb") as file:
                out_dict = pickle.load(file)
            # The sum of weights is stored in the "sumw" key
            # You can access it like this:
            for key in out_dict:
                sumw = next(iter(out_dict[key]["nominal"]["sumw"].values()))
            total_sumw += sumw
    except:
        warnings.warn(
            f"Error loading genweights for dataset: {dataset}. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        total_sumw = 1

    # print(f"Total sum of weights for all pickles for {dataset}: {total_sumw}")
    return total_sumw


def load_samples(
    data_dir: Path,
    samples: dict[str, str],
    columns: list[str],
    region: str,
    extra_columns: dict[str] = None,
    filters: list[tuple[str, str, str]] = None,
    variation: str = None,
    frac: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Load samples from a specified directory and return them as a dictionary.
    :param data_dir: The directory where the datasets are stored.
    :param samples: A dictionary where keys are process names and values are all the datasets corresponding to that process.
    :param columns: A list of columns to load from the datasets.
    :param region: The region to load the parquets from (e.g., "signal-all")
    :param extra_columns: A dictionary where keys are dataset names and values are lists of additional columns to load for that dataset.
    :param filters: A list of filters to apply when loading the datasets.
    :param frac: If >1, for data keep every `frac`-th event; for MC scale event weights by `frac`.
    :return: A dictionary with dataset/sample names as keys and DataFrames as values.
    """
    events_dict = {}
    for process, datasets in samples.items():
        events_list = []
        for dataset in datasets:
            columns_to_load = columns
            if extra_columns and dataset in extra_columns:
                columns_to_load += extra_columns[dataset]

            # Uncomment to debug
            # print(f"Loading dataset: {dataset}")
            # print(f"Looking in: {data_dir / dataset / 'parquet' / f'{region}*.parquet'}")
            # print(list(Path(data_dir / dataset / "parquet").glob(f'{region}*.parquet')))
            # print(f"Columns to load: {columns_to_load}")

            search_path = Path(data_dir / dataset / "parquet" / "nominal" / region)
            if variation:
                search_path = Path(data_dir / dataset /  "parquet" / variation / region)
            print(f"\n[DEBUG] Script is searching in path: {search_path}\n")

            # --- REPLACE THE OLD 'try' BLOCK WITH THIS ---
            try:
                # Use os.listdir() which can be more robust on network filesystems
                if search_path.exists():
                    file_list = [f for f in search_path.iterdir() if f.name.endswith(".parquet")]
                    # print(f"[DEBUG] Found files with os.listdir: {file_list}")
                else:
                    print(f"[DEBUG] Path does not exist: {search_path}")
                    file_list = []

                # If no files were found, skip to the next dataset
                if not file_list:
                    warnings.warn(
                        f"No parquet files found in {search_path}. Skipping dataset {dataset}.",
                        stacklevel=2,
                    )
                    continue

                events = pd.read_parquet(
                    file_list,
                    filters=filters,
                    columns=columns_to_load,
                )
            # --- END REPLACEMENT ---

            # try:
            # Load the dataset into a DataFrame
            #    events = pd.read_parquet(
            #        list(Path(data_dir / dataset / "parquet").glob(f"{region}*.parquet")),
            #        filters=filters,
            #        columns=columns_to_load,
            #    )
            except pa.lib.ArrowInvalid as e:
                warnings.warn(f"ArrowInvalid error: {e}. Skipping dataset {dataset}.", stacklevel=2)
                print("[DEBUG] attempted columns:", columns_to_load)

                parquet_dir = Path(data_dir) / dataset / "parquet" / "nominal" / region
                files = sorted(parquet_dir.glob("*.parquet"))
                print(f"[DEBUG] parquet dir: {parquet_dir}")
                print(f"[DEBUG] n files: {len(files)}")

                if files:
                    import pyarrow.parquet as pq
                    schema_cols = pq.read_schema(files[0]).names
                    missing = sorted(set(columns_to_load) - set(schema_cols))
                    print(f"[DEBUG] first file: {files[0]}")
                    print(f"[DEBUG] missing cols: {missing}")
                    print(f"[DEBUG] all cols: {schema_cols}")
                continue
            except:
                print(f"Error loading dataset: {dataset}. Skipping.")
                print(
                    "List of files available: ",
                    list(Path(data_dir / dataset / "parquet").glob(f"{region}*.parquet")),
                )
                continue

            if "data" not in process:
                # For MC datasets, we need to normalize the weights
                sum_genweights = get_sum_genweights(data_dir, dataset)
                print(f"Using sum_genweights for {dataset}: {sum_genweights}")

                events["weight_nonorm"] = events["weight"]
                events["finalWeight"] = events["weight"] / sum_genweights
                events["sum_genWeight"] = sum_genweights
            else:
                # For data, we just keep the weight as is
                events["weight_nonorm"] = events["weight"]
                events["finalWeight"] = events["weight"]

            # Apply fractional loading / weight scaling if requested
            try:
                if isinstance(frac, int) and frac > 1:
                    if "data" in process:
                        orig_n = len(events)
                        events = events.iloc[::frac].reset_index(drop=True)
                        print(f"[DEBUG] Downsampled data {dataset}: kept 1/{frac} ({len(events)}/{orig_n}) entries")
                    else:
                        events["weight_nonorm"] = events["weight_nonorm"] / frac
                        events["finalWeight"] = events["finalWeight"] / frac
                        print(f"[DEBUG] Scaled MC weights for {dataset} by 1/{frac}")
                elif frac is not None and (not isinstance(frac, int) or frac < 1):
                    warnings.warn(f"Invalid frac={frac}. Expected int>=1. Ignoring.")
            except Exception as e:
                warnings.warn(f"Error applying frac={frac} to dataset {dataset}: {e}")

            # Add the DataFrame to the dictionary with the dataset name as the key
            events_list.append(events)
            print(f"Loaded {dataset: <50}: {len(events)} entries")

        # Combine all DataFrames for the process/sample
        # print(events_list)

        if events_list:
            events_dict[process] = pd.concat(events_list)
        else:
            warnings.warn(
                f"No valid events loaded for process {process}.", category=UserWarning, stacklevel=2
            )

    return events_dict
