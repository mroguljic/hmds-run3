# Datasets lists

## Setup

- Setup rucio (one time at login):
```
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/cms.cern.ch/rucio/setup-py3.sh
voms-proxy-init -voms cms -rfc -valid 192:00
export RUCIO_ACCOUNT=$whoami
```

## Make lists

```
python3 make_filelists.py $VERSION
```

e.g.
```
python3	make_filelists.py v15
```

## Rucio requests

Rucio allows you to transfer CMS datasets.

- To check requests (replace USERNAME):
```
rucio list-rules --account USERNAME
```
e.g., the output is as follows:
```
ID                                ACCOUNT    SCOPE:NAME                                                                                                                                     STATE[OK/REPL/STUCK]    RSE_EXPRESSION    COPIES    SIZE    EXPIRES (UTC)        CREATED (UTC)
--------------------------------  ---------  ---------------------------------------------------------------------------------------------------------------------------------------------  ----------------------  ----------------  --------  ------  -------------------  -------------------
b635b7ef45ab454ba9cbde1d74bd7759  cmantill   cms:/Zto2Nu-2Jets_PTNuNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM  OK[227/0/0]             T1_US_FNAL_Disk   1         N/A     2025-01-21 17:15:03  2024-08-12 16:21:43
```

- To list replicas:
```
rucio list-dataset-replicas cms:/DATASET
```

- Make request (replace DATASET and lifetime e.g. 14000000), to transfer to T1_US_FNAL_Disk
```
rucio add-rule cms:/DATASET 1 T1_US_FNAL_Disk --activity "User AutoApprove" --lifetime [# of seconds] --ask-approval --comment ''
```
e.g.:
```
rucio add-rule "cms:/Zto2Nu-2Jets_PTNuNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM" 1 T1_US_FNAL_Disk --activity "User AutoApprove" --lifetime 14000000 --ask-approval --comment ''
```
