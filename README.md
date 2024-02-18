# EEG-Conformer

> [!Note]
> this repo is based on: https://github.com/eeyhsong/EEG-Conformer

We made some modifications on the original model to handle our 4-class online/offline MI task, the Dataset is [lyh-Dataset](https://github.com/BCI-NJU/lyh-Dataset), which is recorded and made by ourselves. The 4 task involves:
- Left-hand
- Right-hand
- Legs
- Nothing(can be regarded as the *Others* state)

## Usage

```
cd ./EEG-Conformer
python ./Codes/conformer.py
```

## Filepath Description

```
.
├── Codes: core codings
│   ├── conformer.py: the wrapper file/entry of the programs
│   ├── evaluation.py
│   ├── ExP.py: single subject training procedure
│   ├── models.py
│   ├── playground.ipynb: temporary files for us to play with
│   └── test.py
├── Datasets
│   ├── BCIC42a
│   └── lyh_dataset
├── Deprecated_codes: codes no longer used
│   ├── base.py
│   ├── BCIC42a-sub1.ipynb
│   └── eegconformer.py: another version of conformer implementation from BrainDecode
├── Models: best model state dict
│   ├── BCIC42a-sub1
│   └── lyh_dataset
├── README.md
└── Results: training history, config, acc, loss
    ├── 2024-02-18 08:17:42.851201
    └── 2024-02-18 08:29:27.040304
```

## Requirements

- we run the programs with python 3.7 version, higher version of python untested.



