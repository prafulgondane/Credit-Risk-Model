import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import joblib
import pickle
import random
import numpy as np
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import gc
import pickle

print("Starting")

#you can load from disk
X_train_with_encoded_feat_df, y_train = pickle.load(open('train_data.pkl', 'rb')) 
X_cv_with_encoded_feat_df, y_cv = pickle.load(open('cv_data.pkl', 'rb'))
X_test_with_encoded_feat_df, SK_ID_CURR_test = pickle.load(open('test_data.pkl', 'rb'))

#X_train_with_encoded_feat_df['TARGET'] = y_train.values.tolist()
#X_cv_with_encoded_feat_df['TARGET'] = y_cv.values.tolist()

# pd.concat([X_train_with_encoded_feat_df, X_cv_with_encoded_feat_df], ignore_index=True)

# Train
#X_train_with_encoded_feat_df = pd.concat([X_train_with_encoded_feat_df, X_cv_with_encoded_feat_df], ignore_index=True)
X_train_with_encoded_feat_df = pd.read_csv('LGBM_FINAL_DATA.csv')
y_train = X_train_with_encoded_feat_df['TARGET']
X_train = X_train_with_encoded_feat_df.drop(columns=['TARGET'])

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, stratify = y_train, test_size=0.2, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify = y_temp, test_size=0.5, random_state=42)

X_test = X_test_with_encoded_feat_df

print("x_train, y_train :", X_train.shape, y_train.shape)
print("X_cv, y_test :", X_cv.shape, y_cv.shape)
print("x_test, SK_ID_CURR_test :", X_test.shape, SK_ID_CURR_test.shape)
gc.collect()

# Create the model
model = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )
    
        
# Train the model
model.fit(X_train, y_train, eval_metric = 'auc',
                eval_set = [(X_cv, y_cv), (X_train, y_train)],
                eval_names = ['valid', 'train'],
                early_stopping_rounds = 100, verbose = 100)

train_preds = model.predict_proba(X_train)[:, 1]
roc = roc_auc_score(y_train, train_preds)
print(' AUC - ROC TRAIN score %.6f' % roc)    

cv_preds = model.predict_proba(X_cv)[:, 1]
roc = roc_auc_score(y_cv, cv_preds)
print(' AUC - ROC CV score %.6f' % roc)   

test_preds = model.predict_proba(X_test)[:, 1] 

    # fold_importance_df = pd.DataFrame()
    # fold_importance_df["feature"] = feats
    # fold_importance_df["importance"] = model.feature_importances_
    # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    # print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_labels, oof_preds[valid_indices])))
# roc = roc_auc_score(valid_labels, oof_preds[valid_indices])
# print(' AUC - ROC score %.6f' % roc)
print('Saving Best Model')
pickle.dump(model, open('model_p.pkl', 'wb'))
joblib.dump(model, 'model_j.pkl')




#data = pd.read_csv('iris.csv')
##print(data.head())
#X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
#Y = data['species']
#clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
#clf.fit(X, Y)
#joblib.dump(clf, 'model.pkl')
