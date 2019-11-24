# RSNA Intracranial Hemorrhage Detection

- This is the project for [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) hosted on Kaggle in 2019.
- It ended up at [11th place](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/leaderboard) in the competition.


## Table of Contents

*   [Directory Layout](#directory-layout)
*   [Solution Overview](#solution-overview)
*   [How to Run](#how-to-run)
    *   [Requirements](#requirements)
    *   [Preprocessing](#preprocessing)
    *   [Training](#training)
    *   [Predicting](#predicting)
    *   [Second Level Model](#second-level-model-how-to-run)
    *   [Ensembling](#ensembling)
*   [Download](#download)
    *   [Trained Weights](#trained-weights)
*   [License](#license)


## Directory layout

```
.
├── bin           # Scripts to perform various tasks such as `preprocess`, `train`.
├── cache         # Where preprocessed outputs are saved.
├── conf          # Configuration files for classification models.
├── input         # Input files provided by kaggle. 
├── model         # Where classification model outputs are saved.
├── meta          # Where second level model outputs are saved.
├── src           # 
└── submission    # Where submission files are saved.
```

Missing directories will be created when `./bin/preprocess.sh` is run.


## Solution Overview

You can find it on kaggle forum.

- https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117330


## How to run

Please put `./input` directory in the root level and unzip the downloaded file from [kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data) there. The zipped file has to be the one provided for 2nd stage and the file size should be 180GB before unzipping.

Please make sure you run each of the scripts from parent directory of `./bin`.


### Requirements

The library versions we used. It does not mean other versions can not be used but not tested.

- Python 3.6.6
- CUDA 10.0 (CUDA driver 410.79)
- [Pytorch](https://pytorch.org/) 1.1.0
- [NVIDIA apex](https://github.com/NVIDIA/apex) 0.1 (for mixed precision training)


### Preprocessing

~~~
$ sh ./bin/preprocess.sh
~~~

[preprocess.sh](./bin/preprocess.sh) does the following at once.

- Creates directories such as `./cache`, `./model` if needed.
- [dicom_to_dataframe.py](./src/preprocess/dicom_to_dataframe.py) reads dicom files and save its metadata into the dataframe.
- [create_dataset.py](./src/preprocess/create_dataset.py) creates a dataset for train/test.
- [make_folds.py](./src/preprocess/make_folds.py) makes folds(n=8) for cross validation. 


### Training (classification model)

~~~
$ sh ./bin/train.sh
~~~

- Trains two types of models `se_resnext50_32x4d` and `se_resnext101_32x4d` with 8 folds each. 


### Predicting (classification model)

~~~
$ sh ./bin/predict.sh
~~~

- Makes predictions for validation data (out-of-fold predictions).
- Makes predictions for test data.
- Checkpoints from 2nd and 3rd epoch of each fold are used for predictions.


### Second level model

~~~
$ sh ./bin/predict_meta.sh
~~~

- Ensembles out-of-fold predictions from the previous step (used as meta features to construct train data).
- Ensembles test predictions from the previous step (used as meta features to construct test data).
- Trains `LightGBM`, `Catboost` and `XGB` with 8 folds each.
- Predicts on test data using each of the trained models.


### Ensembling (+postprocessing)

~~~
$ sh ./bin/ensemble.sh
~~~

- Ensembles predictions from the previous step.
- Makes a submission file.


## Download


### Trained Weights

Due to kaggle dataset limit, model110 checkpoints are split into two parts.
To use these checkpoints, please download them and unzip at `./model` directory. You can skip `Training` phase and start `Predicting` by using them.

- model100 https://www.kaggle.com/appian/rsna-model100
- model110 (part 1) https://www.kaggle.com/appian/rsna-model110-1
- model110 (part 2) https://www.kaggle.com/appian/rsna-model110-2


## License

The license is MIT.
