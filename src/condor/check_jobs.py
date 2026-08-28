"""
Checks that there is an output for each job submitted.

Author: Raghav Kansal
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import submit
from colorama import Fore, Style

from hbb import run_utils


def print_red(s):
    return print(f"{Fore.RED}{s}{Style.RESET_ALL}")


def main(args):
    eosdir = f"{args.location}/{args.tag}/{args.year}/"
    samples = Path(eosdir).iterdir()

    jdls = [jdl for jdl in Path(f"condor/{args.tag}/").iterdir() if jdl.suffix == ".jdl"]

    jdl_dict = {}
    for sample in samples:
        # extract the sample name
        s = str(sample).split("/")[-1]

        x = [
            int(str(jdl)[:-4].split("_")[-1])
            for jdl in jdls
            if str(jdl).split("_")[1].split("/")[-1] == args.year
            and "_".join(str(jdl).split("_")[2:-1]) == s
        ]
        # get number of jdl (submission files)
        if len(x) > 0:
            jdl_dict[s] = np.sort(x)[-1] + 1

    running_jobs = []
    if args.check_running:
        os.system("condor_q | awk '{print $9}' > running_jobs.txt")
        with Path("running_jobs.txt").open() as f:
            lines = f.readlines()

        running_jobs = [s[:-4] for s in lines if s.endswith(".sh\n")]

    missing_files = []
    err_files = []
    # job ids to re-run, keyed by subsample, so they can be submitted in one
    # condor_submit per subsample rather than one per job
    missing_by_sample = {}

    for sample, n_jdls in jdl_dict.items():
        print(f"Checking {sample}")

        ### CHECK PARQUETS
        if not Path(f"{eosdir}/{sample}/parquet").exists():
            print_red(f"No parquet directory for {sample}!")

            # if no parquet directory exists, append all possible files
            for i in range(n_jdls):
                if f"{args.year}_{sample}_{i}" in running_jobs:
                    print(f"Job #{i} for sample {sample} is running.")
                    continue

                jdl_file = f"condor/{args.tag}/{args.year}_{sample}_{i}.jdl"
                err_file = f"condor/{args.tag}/logs/{args.year}_{sample}_{i}.err"
                missing_files.append(jdl_file)
                err_files.append(err_file)
                missing_by_sample.setdefault(sample, []).append(i)

            # and go to the next sample
            continue

        outs_parquet = [
            int(out.stem.removeprefix("part"))
            for out in Path(f"{eosdir}/{sample}/parquet").rglob("part*.parquet")
        ]
        # print(f"Out parquets: {outs_parquet}")

        if not Path(f"{eosdir}/{sample}/pickles").exists():
            print_red(f"No pickles directory for {sample}!")
            continue

        ### CHECK PICKLES
        outs_pickles = [
            int(str(out).split(".")[0].split("_")[-1])
            for out in Path(f"{eosdir}/{sample}/pickles").glob("*.pkl")
        ]

        # print(f"Out pickles: {outs_pickles}")

        for i in range(jdl_dict[sample]):
            if i not in outs_pickles:
                if f"{args.year}_{sample}_{i}" in running_jobs:
                    print(f"Job #{i} for sample {sample} is running.")
                    continue

                print_red(f"Missing output pickle #{i} for sample {sample}")
                jdl_file = f"condor/{args.tag}/{args.year}_{sample}_{i}.jdl"
                err_file = f"condor/{args.tag}/logs/{args.year}_{sample}_{i}.err"
                missing_files.append(jdl_file)
                err_files.append(err_file)
                missing_by_sample.setdefault(sample, []).append(i)

            if i not in outs_parquet and i in outs_pickles:
                print_red(
                    f"Missing output parquet #{i} for sample {sample} but pickle file produced"
                )
                jdl_file = f"condor/{args.tag}/{args.year}_{sample}_{i}.jdl"
                err_file = f"condor/{args.tag}/logs/{args.year}_{sample}_{i}.err"
                missing_files.append(jdl_file)
                err_files.append(err_file)
                missing_by_sample.setdefault(sample, []).append(i)

    if args.submit_missing and missing_by_sample:
        batch_dir = Path(f"condor/{args.tag}/batch_resubmit")
        batch_dir.mkdir(parents=True, exist_ok=True)
        proxy = os.environ["X509_USER_PROXY"]

        for sample, jobids in missing_by_sample.items():
            prefix = f"{args.year}_{sample}"
            batch_condor = f"{batch_dir}/{prefix}.jdl"
            submit.write_template(
                "src/condor/submit.templ.batch.jdl",
                batch_condor,
                {
                    "dir": f"condor/{args.tag}",
                    "prefix": prefix,
                    "proxy": proxy,
                    "jobids": " ".join(str(j) for j in sorted(set(jobids))),
                },
            )
            print(f"Resubmitting {len(set(jobids))} jobs for {sample}")
            os.system(f"condor_submit {batch_condor}")

    print(f"{len(missing_files)} files to re-run:")
    for f in missing_files:
        print(f)

    print("\nError files:")
    for f in err_files:
        print(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--location",
        required=True,
        help="output folder of jobs e.g. /eos/uscms/store/user/lpchbbrun3/cmantill/",
        type=str,
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="tag e.g. 25Jun25_v12",
        type=str,
    )
    parser.add_argument("--year", help="year", type=str)
    run_utils.add_bool_arg(parser, "submit-missing", default=False, help="submit missing files")
    run_utils.add_bool_arg(
        parser,
        "check-running",
        default=False,
        help="check against running jobs as well (running_jobs.txt will be updated automatically)",
    )

    args = parser.parse_args()
    main(args)
