python -m src.postprocess.make_submission --inputs "['./meta/meta100_lgb.pkl', './meta/meta100_cat.pkl', './meta/meta100_xgb.pkl']" --output ./submission/sub001.csv

#kaggle competitions submit rsna-intracranial-hemorrhage-detection -m "" -f ./submission/sub001.csv
