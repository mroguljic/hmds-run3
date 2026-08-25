"""
Runs a coffea processors on a single file or a set of files (ONLY for testing)
"""

from __future__ import annotations

import argparse
import logging
import pickle
import shutil
from pathlib import Path

import awkward as ak
import dask
import uproot
import yaml
from coffea import nanoevents
from coffea.dataset_tools import apply_to_fileset, max_chunks, preprocess

from hbb.common_vars import DATA_SAMPLES
from hbb.run_utils import get_dataset_spec, get_fileset
from hbb.xsecs import xsecs


def is_data(dataset: str) -> bool:
    return any(sample in dataset for sample in DATA_SAMPLES)


def nfiles_per_dataset(spec: dict) -> dict:
    return {dataset: len(entry.get("files") or {}) for dataset, entry in spec.items()}


def check_complete(dataset: str, msg: str):
    # Skipped events renormalize away for MC (sum_genweights), but are lost lumi for data.
    if is_data(dataset):
        raise RuntimeError(msg)
    logging.warning("%s", msg)


def run(year: str, fileset: dict, args: argparse.Namespace):
    """Run processor without fancy dask (outputs then need to be accumulated manually)"""

    local_dir = Path().resolve()

    if args.save_skim or args.save_skim_nosysts:
        # intermediate files are stored in the "./outparquet" local directory
        local_parquet_dir = local_dir / "outparquet"
        if local_parquet_dir.is_dir():
            shutil.rmtree(local_parquet_dir)
        local_parquet_dir.mkdir(parents=True, exist_ok=True)

    uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

    # if the fileset is empty, use dummy file to check if the processor works
    if not fileset:
        dataset = "GluGlu_Hto2B"
        fname = "root://cmsxrootd.fnal.gov//store/mc/Run3Summer22NanoAODv12/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/50000/3a484476-0efd-469c-b376-c09628e3d380.root"
        dict_process_files = {
            dataset: {
                "files": {fname: "Events"},
                "metadata": {
                    "dataset": dataset,
                },
            }
        }
        fileset = {fname: "Events"}
        events = nanoevents.NanoEventsFactory.from_root(
            fileset,
            schemaclass=nanoevents.NanoAODSchema,
            metadata={"dataset": "test"},
        ).events()
        print(events.metadata)
        # Uncomment the following lines to run the processor directly on the events
        # out = p.process(events)
        # (computed,) = dask.compute(out)
    else:
        dict_process_files = get_dataset_spec(fileset)

    # Use preprocess from coffea
    preprocessed_available, preprocessed_total = preprocess(
        dict_process_files,
        align_clusters=True,
        skip_bad_files=True,
        recalculate_steps=False,
        files_per_batch=1,
        file_exceptions=(OSError,),
        step_size=20_000,
        save_form=False,
        uproot_options={
            "xrootd_handler": uproot.source.xrootd.MultithreadedXRootDSource,
            "allow_read_errors_with_report": True,
        },
        step_size_safety_factor=0.5,
    )
    print(
        "Number of files preprocessed: ",
        len(preprocessed_available),
        " out of ",
        len(preprocessed_total),
    )

    nfiles_avail = nfiles_per_dataset(preprocessed_available)
    for dataset, ntotal in nfiles_per_dataset(preprocessed_total).items():
        ndropped = ntotal - nfiles_avail.get(dataset, 0)
        if ndropped:
            check_complete(dataset, f"{dataset}: {ndropped}/{ntotal} files unreadable")

    # TODO: customize processor
    from hbb.processors import categorizer

    p = categorizer(
        xsecs=xsecs,
        year=year,
        nano_version=args.nano_version,
        save_skim=args.save_skim,
        evaluate_BDT=args.BDT,
        skim_outpath="outparquet",
        btag_eff=args.btag_eff,
        save_skim_nosysts=args.save_skim_nosysts,
    )

    fileset_to_run = preprocessed_available
    if args.max_chunks:
        for dataset in fileset_to_run:
            if is_data(dataset):
                msg = f"--max-chunks would discard luminosity from {dataset}; MC only"
                raise ValueError(msg)
        fileset_to_run = max_chunks(preprocessed_available, args.max_chunks)

    full_tg, rep = apply_to_fileset(
        data_manipulation=p,
        fileset=fileset_to_run,
        schemaclass=nanoevents.NanoAODSchema,
        uproot_options={
            "allow_read_errors_with_report": (OSError, KeyError),
            "xrootd_handler": uproot.source.xrootd.MultithreadedXRootDSource,
            "timeout": 1800,
        },
    )
    output, report = dask.compute(full_tg, rep)

    for dataset, entry in (report or {}).items():
        exceptions = getattr(entry, "exception", None)
        nfailed = 0 if exceptions is None else sum(e is not None for e in ak.to_list(exceptions))
        if nfailed:
            check_complete(dataset, f"{dataset}: {nfailed} chunks failed to read")
    # print("output ", output)

    # save the output to a pickle file
    with Path(f"{local_dir}/{args.starti}-{args.endi}.pkl").open("wb") as f:
        pickle.dump(output, f)
    print("Saved output to ", f"{local_dir}/{args.starti}-{args.endi}.pkl")

    # COMBINE FILES
    # otherwise it will complain about too many small files
    # This is the CORRECTED version of the file-combining block for run.py

    if args.save_skim or args.save_skim_nosysts:

        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        jer_vars = []
        for entry in Path(local_parquet_dir).iterdir():
            if entry.is_dir():
                jer_vars.append(entry.name)

        # compile parquet files from each jer_var/region/ directory
        # save as {jer_var}_{region_name}.parquet for easy transfer
        for local_var in jer_vars:
            # only find subfolders with parquet files
            parquet_folders = set()
            for parquet_file in Path(local_parquet_dir / local_var).rglob("*.parquet"):
                parquet_folders.add(str(parquet_file.parent.resolve()))

            for folder in parquet_folders:
                full_path = Path(folder)
                # This is the simpler, correct way to get the region name
                region_name = full_path.name
                pddf = pd.read_parquet(folder)

                table = pa.Table.from_pandas(pddf)
                # This saves the combined file as {local_var}_{region_name}.parquet locally
                output_file = f"{local_dir}/{local_var}_{region_name}.parquet"
                pq.write_table(table, output_file)
                print("Saved parquet file to ", output_file)

        # remove subfolder
        print("Removing temporary folder: ", local_parquet_dir)
        shutil.rmtree(local_parquet_dir)


def main(args):

    print(args)

    if len(args.files):
        fileset = {f"{args.year}_{args.files_name}": args.files}
    else:
        if args.yaml:
            with Path(args.yaml).open() as file:
                samples_to_submit = yaml.safe_load(file)
            try:
                samples_to_submit = samples_to_submit[args.year]
            except Exception as e:
                raise KeyError(f"Year {args.year} not present in yaml dictionary") from e

            samples = samples_to_submit.keys()
            subsamples = []
            for sample in samples:
                subsamples.extend(samples_to_submit[sample].get("subsamples", []))
        else:
            samples = args.samples
            subsamples = args.subsamples

        fileset = get_fileset(
            args.year,
            args.nano_version,
            samples,
            subsamples,
            args.starti,
            args.endi,
        )

    print(f"Running on fileset {fileset}")
    run(args.year, fileset, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        default="2024",
        choices=["2022", "2022EE", "2023", "2023BPix", "2024"],
    )
    parser.add_argument("--starti", default=0, help="start index of files", type=int)
    parser.add_argument(
        "--max-chunks", default=None, help="MC debugging only: chunks per dataset", type=int
    )
    parser.add_argument("--endi", default=-1, help="end index of files", type=int)
    parser.add_argument(
        "--samples",
        default=[],
        help="which samples to run",  # , default will be all samples",
        nargs="*",
    )
    parser.add_argument(
        "--subsamples",
        default=[],
        help="which subsamples, by default will be all in the specified sample(s)",
        nargs="*",
    )
    parser.add_argument(
        "--nano-version",
        type=str,
        default="v15",
        choices=["v12", "v12v2_private", "v14_private", "v15", "v15_private"],
        help="NanoAOD version",
    )
    parser.add_argument(
        "--files", default=[], help="set of files to run on instead of samples", nargs="*"
    )
    parser.add_argument(
        "--files-name",
        type=str,
        default="files",
        help="sample name of files being run on, if --files option used",
    )
    parser.add_argument(
        "--yaml", default=None, help="yaml file with samples and subsamples", type=str
    )
    parser.add_argument(
        "--BDT",
        action="store_true",
        help="Evaluate BDT scores and use for categorization",
        default=False,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--save-skim",
        action="store_true",
        help="save skimmed (flat ntuple) files",
        default=False,
    )
    group.add_argument(
        "--btag-eff",
        action="store_true",
        help="compute b-tag efficiencies for mc",
        default=False,
    )
    group.add_argument(
        "--save-skim-nosysts",
        action="store_true",
        help="save skimmed files, skip systematics",
        default=False,
    )

    args = parser.parse_args()

    main(args)
