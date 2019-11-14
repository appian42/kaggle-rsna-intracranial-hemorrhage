from pprint import pprint

import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, log_loss

from ..utils import mappings


_params_lgb = {
    'objective': 'binary',
    #"metric": 'auc',
    'num_leaves': 30,
    'min_child_samples': 5,
    'max_depth': 5,
    'learning_rate': 0.02,
    "boosting_type": "gbdt",
    "bagging_freq": 2,
    "bagging_fraction": 0.7,
    "bagging_seed": 11,
    "verbosity": -1,
    'reg_alpha': 0.9,
    'reg_lambda': 0.9,
    'colsample_bytree': 0.9,
    'importance_type': 'gain',
    'random_seed': 20,
    'n_estimators': 5000,
    'n_jobs': 6,
}

_params_catboost = {
    #'task_type': 'GPU',
    'depth': 4,
    'learning_rate': 0.02,
    #'l2_leaf_reg': 88,
    #'random_strength': 0,
    #'bagging_temperature': 0,
    'random_seed': 42,
    'early_stopping_rounds': 100,
    'iterations': 20000,
    'thread_count': 8,    
    #'od_type': 'Iter',
}

_params_xgb = {
    'nthread': 8,
    'learning_rate': 0.08,
    'max_depth': 5,
    'colsample_bytree': 0.9,
    'subsample': 0.9,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'predictor': 'gpu_predictor',
    'silent': True,
    #'labmda': 0.3,
    #'alpha': 0.3,
}

_columns = [
    'any', 'epidural', 'subdural', 'subarachnoid', 'intraventricular', 'intraparenchymal',
    'n_slice', 'pos', 'mean_by_study',
    'brain_ratio', 
]


def run(train_df, test_df, gb_type, eps=1e-6):

    logloss_all = []
    oof_all = []
    test_all = []

    for i_label in range(0,6):

        n_fold = int(train_df.fold.max()+1)
        oof = np.zeros(len(train_df))
        test = np.zeros(len(test_df))

        label = mappings.num_to_label[i_label]
        gt_label = 'gt_%s' % label

        left_preds = ['%s_l%d' % (label, i) for i in range(1, 10)]
        right_preds = ['%s_r%d' % (label, i) for i in range(1, 10)]
        columns = left_preds + right_preds + _columns 

        X_test = test_df[columns]

        for i_fold in range(n_fold):

            print('\n----- fold:%d label:%s(%d) -----' % (i_fold, label, i_label))

            _train = train_df[train_df.fold != i_fold]
            _valid = train_df[train_df.fold == i_fold]

            X_train = _train[columns]
            y_train = _train[gt_label]
            X_valid = _valid[columns]
            y_valid = _valid[gt_label]

            if gb_type == 'lgb':
                model = lgb.LGBMClassifier(**_params_lgb)
                model.fit(
                    X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], 
                    eval_metric='multi_logloss',
                    verbose=50,
                    early_stopping_rounds=100,
                )
                pred_valid = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:,1]
                pred_test = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:,1]

            elif gb_type == 'cat':
                model = CatBoostClassifier(**_params_catboost, eval_metric='Logloss', loss_function='Logloss')
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], cat_features=[], use_best_model=True, verbose=100)
                pred_valid = model.predict_proba(X_valid)[:,1]
                pred_test = model.predict_proba(X_test)[:,1]
        
            elif gb_type == 'xgb':
                train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)
                valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)
                watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
                model = xgb.train(dtrain=train_data, evals=watchlist, params=_params_xgb, verbose_eval=50, num_boost_round=500, early_stopping_rounds=50)
                pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_valid.columns), ntree_limit=model.best_ntree_limit)
                pred_test = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

            oof[_valid.index] = pred_valid
            test += pred_test

            logloss = log_loss(y_valid, np.clip(pred_valid, eps, 1-eps))

            print('logloss', logloss)

        logloss = log_loss(train_df[gt_label], oof)
        print('logloss(all)', logloss)

        test /= n_fold

        oof_all.append(oof)
        test_all.append(test)
        logloss_all.append(logloss)

    pprint(logloss_all)

    return oof_all, test_all, logloss_all
