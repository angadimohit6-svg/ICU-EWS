import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from icu_ews.ml import artifacts, features as F
from icu_ews.ml.training import _threshold_from_oof
from sklearn.model_selection import cross_val_predict

frames = artifacts.load_frames("demo-cohort-v1")
X, y, meta = F.training_frame(frames["patients"], frames["observations"], frames["labs"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

candidates = [
    dict(n_estimators=450, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0, scale_pos_weight=3),
    dict(n_estimators=600, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.7, min_child_weight=1, reg_lambda=1.0, scale_pos_weight=5),
    dict(n_estimators=400, max_depth=4, learning_rate=0.06, subsample=0.9, colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.5, scale_pos_weight=4),
    dict(n_estimators=700, max_depth=2, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0, scale_pos_weight=3),
    dict(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.9, colsample_bytree=0.7, min_child_weight=4, reg_lambda=2.0, scale_pos_weight=2),
]
for i, params in enumerate(candidates):
    model = xgb.XGBClassifier(**params, random_state=42, n_jobs=4, eval_metric="logloss", tree_method="hist")
    scores = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc")
    ap = cross_val_score(model, Xtr, ytr, cv=cv, scoring="average_precision")
    model.fit(Xtr, ytr)
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    p = model.predict_proba(Xte)[:, 1]
    print(i, f"cvAUC {scores.mean():.3f}±{scores.std():.3f} cvAP {ap.mean():.3f} | test AUC {roc_auc_score(yte,p):.3f} AP {average_precision_score(yte,p):.3f} Brier {brier_score_loss(yte,p):.3f}")
