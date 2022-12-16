test100="[\
    ['./model/model100/fold0_ep2_test_tta5.pkl', './model/model100/fold0_ep3_test_tta5.pkl'],\
    ['./model/model100/fold1_ep2_test_tta5.pkl', './model/model100/fold1_ep3_test_tta5.pkl'],\
    ['./model/model100/fold2_ep2_test_tta5.pkl', './model/model100/fold2_ep3_test_tta5.pkl'],\
    ['./model/model100/fold3_ep2_test_tta5.pkl', './model/model100/fold3_ep3_test_tta5.pkl'],\
    ['./model/model100/fold4_ep2_test_tta5.pkl', './model/model100/fold4_ep3_test_tta5.pkl'],\
    ['./model/model100/fold5_ep2_test_tta5.pkl', './model/model100/fold5_ep3_test_tta5.pkl'],\
    ['./model/model100/fold6_ep2_test_tta5.pkl', './model/model100/fold6_ep3_test_tta5.pkl'],\
    ['./model/model100/fold7_ep2_test_tta5.pkl', './model/model100/fold7_ep3_test_tta5.pkl'],\
]"

python -m src.postprocess.ensemble --inputs "${test100}" --output ./input/test_predictions.csv

# python -m src.postprocess.make_submission --inputs "['./meta/meta100_lgb.pkl', './meta/meta100_cat.pkl', './meta/meta100_xgb.pkl']" --output ./submission/sub001.csv
#kaggle competitions submit rsna-intracranial-hemorrhage-detection -m "" -f ./submission/sub001.csv
