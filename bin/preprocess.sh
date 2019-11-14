mkdir -p cache model meta submission

# train
python -m src.preprocess.dicom_to_dataframe --input ./input/stage_2_train.csv --output ./cache/train_raw.pkl --imgdir ./input/stage_2_train_images
python -m src.preprocess.create_dataset --input ./cache/train_raw.pkl --output ./cache/train.pkl --brain-diff 60
python -m src.preprocess.make_folds --input ./cache/train.pkl --output ./cache/train_folds8_seed300.pkl --n-fold 8 --seed 300

# test 
python -m src.preprocess.dicom_to_dataframe --input ./input/stage_2_sample_submission.csv --output ./cache/test_raw.pkl --imgdir ./input/stage_2_test_images
python -m src.preprocess.create_dataset --input ./cache/test_raw.pkl --output ./cache/test.pkl --brain-diff 60
