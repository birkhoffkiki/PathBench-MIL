# PathBench-MIL
The downstream evaluation code for the PathBench.
## WSI Classification
The code for WSI Classification tasks.
1. Process your WSI data into patches, extract features using a pre-trained model, and organize them as following:
```bash
DATA_ROOT/
    └── H1/
        └── pt_files/
            ├── resnet50/
            │   ├── slide_1.pt
            │   ├── slide_2.pt
            │   └── ...
            ├── virchow2/
            │   ├── slide_1.pt
            │   ├── slide_2.pt
            │   └── ...
            └── ...
        └── patches/
            ├── slide_1.h5
            ├── slide_2.h5
            └── ...
    └── H2/
        └── ...
    └── H3/
        └── ...
    └── H4/
        └── ...
```
2. Fill the path of your data into `classification/splits/datasets.xlsx`
3. Split your data into training, validation, test, and external (optional) sets following the example in `classification/splits/task1_split.xlsx`
4. Specify the hyper-parameters in `classification/run.sh`
5. Benchmark foundation models by command:
```bash
cd classification
bash run.sh
```

## Survival Prediction
The code for Survival Prediction tasks.

1. Organize your data into an excel file following the example `survival/splits/example.xlsx`
2. Specify the hyper-parameters in `survival/run.sh`
3. run by command
```bash
cd survival
bash run.sh
```

