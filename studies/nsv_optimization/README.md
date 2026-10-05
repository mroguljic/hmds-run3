# nSV definition and cut optimization

`nsv_study.py` studies the secondary-vertex (SV) definition behind
nSV, the nSV threshold, and the P(QCD) cut. It runs on **NanoAOD** (the skims only store the final
nSV count, not per-SV quantities).

Chosen working point: SVs with **ntracks > 2 and 3D decay-length significance > 4**,
**nSV > 4**, **P(QCD) < 0.075**.

## Usage (from the repo root; needs a grid proxy)

```bash
python studies/nsv_optimization/nsv_study.py all --outdir studies/nsv_optimization/output/<YYMMDD>
```

- `dump`: NanoAOD -> reduced parquet (`<outdir>/dump/`). ~15 min for signal + 20 files per QCD HT
  bin (>= 400 GeV) + ttbar on 6 workers; done files are skipped, so rerunning retries xrootd failures.
- `analyze`: -> `<outdir>/plots/` (~2 min). `--chosen "def:k:pqcd"` and `--min-bkg-per-ptbin`
  (default 500) set the working point and the background requirement; `-h` lists the rest.
- `all`: both.