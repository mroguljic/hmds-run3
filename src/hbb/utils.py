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
import pyarrow.parquet as pq
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

    # Load the genweights from the pickle files.
    pickle_files = list(Path(data_dir / dataset / "pickles").glob("*.pkl"))
    if not pickle_files:
        raise FileNotFoundError(
            f"No pickle files found for dataset '{dataset}' in "
            f"{data_dir / dataset / 'pickles'}; cannot compute sum of genweights."
        )

    for pickle_file in pickle_files:
        with Path(pickle_file).open("rb") as file:
            out_dict = pickle.load(file)
        # The sum of weights is stored under out_dict[dataset]["nominal"]["sumw"].
        for key in out_dict:
            sumw = next(iter(out_dict[key]["nominal"]["sumw"].values()))
        total_sumw += sumw

    if total_sumw == 0:
        raise ValueError(
            f"Sum of genweights for dataset '{dataset}' is 0; refusing to normalize "
            "by 0 (or fall back to 1). Check the pickle 'sumw' contents."
        )

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
    prescale: int = 1,
    file_batch: tuple[int, int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load samples from a specified directory and return them as a dictionary.
    :param data_dir: The directory where the datasets are stored.
    :param samples: A dictionary where keys are process names and values are all the datasets corresponding to that process.
    :param columns: A list of columns to load from the datasets.
    :param region: The region to load the parquets from (e.g., "signal-all")
    :param extra_columns: A dictionary where keys are dataset names and values are lists of additional columns to load for that dataset.
    :param filters: A list of filters to apply when loading the datasets.
    :param prescale: If >1, keep only data events with `event % prescale == 0` (a fixed, reproducible
        subset independent of file/chunk ordering, for blinding). MC rows are never dropped, but MC
        weights are scaled down by 1/prescale to match, so Data/MC stays interpretable while blinded.
    :param file_batch: Optional (i_batch, n_batches). Loads only every n_batches-th parquet file,
        starting at i_batch, instead of the whole dataset. Callers that fill histograms can loop
        over batches to keep peak memory at ~1/n_batches of the full dataset - the big QCD samples
        are tens of millions of rows and don't fit in RAM in one piece on a shared node.
        Weights/prescaling are per-row, so batching does not change the summed result.
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

                if file_batch is not None:
                    i_batch, n_batches = file_batch
                    # Sort first so the batch split is deterministic (iterdir order is not).
                    file_list = sorted(file_list)[i_batch::n_batches]
                    if not file_list:
                        # Fewer files than batches - this batch is legitimately empty.
                        continue

                is_data = "data" in process
                has_event_col = False
                if is_data and prescale > 1:
                    has_event_col = "event" in pq.read_schema(file_list[0]).names
                    if has_event_col and "event" not in columns_to_load:
                        columns_to_load = [*columns_to_load, "event"]

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
                print("List of columns attempted to load: ", columns_to_load)
                print(
                    "List of files available: ",
                    list(Path(data_dir / dataset / "parquet").glob(f"{region}*.parquet")),
                )
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

                if prescale > 1:
                    # MC isn't row-subsampled (that would just add unnecessary statistical
                    # noise to the prediction) - instead its normalization is scaled down by
                    # the same factor data was prescaled by, so Data/MC stays interpretable
                    # while blinded (otherwise Data/MC would trivially sit at ~1/prescale).
                    events["weight_nonorm"] = events["weight_nonorm"] / prescale
                    events["finalWeight"] = events["finalWeight"] / prescale
            else:
                # For data, we just keep the weight as is
                events["weight_nonorm"] = events["weight"]
                events["finalWeight"] = events["weight"]

                if prescale > 1:
                    orig_n = len(events)
                    if has_event_col:
                        events = events[events["event"] % prescale == 0].reset_index(drop=True)
                        print(
                            f"[DEBUG] Prescaled data {dataset}: kept 1/{prescale} "
                            f"({len(events)}/{orig_n}) entries via event % {prescale} == 0"
                        )
                    else:
                        # TEMPORARY fallback: 260710 skims predates the 'event'
                        # branch being saved, so we can't prescale by event number. Falls back
                        # to positional row-based prescaling, which is NOT stable across
                        # reskims/reprocessing - only use until data is reskimmed.
                        warnings.warn(
                            f"'event' branch not found in skim for {dataset} - falling back to "
                            f"positional prescale (iloc[::{prescale}]), which is not reproducible "
                            "across reskims. Reskim data to get event-number-based prescaling. ",
                            stacklevel=2,
                        )
                        events = events.iloc[::prescale].reset_index(drop=True)
                        print(
                            f"[DEBUG] Prescaled data {dataset} (positional fallback): kept "
                            f"1/{prescale} ({len(events)}/{orig_n}) entries"
                        )

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
