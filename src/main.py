import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from src.hyper_params import HP
from src.preprocessors import PP


def main():
    all_runs = pd.read_csv(os.path.join(os.path.dirname(__file__), os.pardir, "runs", "runs.csv"))
    for task in all_runs.itertuples():
        if np.isnan(task.roc):
            t0 = time.time()
            score = one_run(task.preproc, task.algo)
            t1 = time.time()
            all_runs.loc[task.Index, "roc"] = score
            all_runs.loc[task.Index, "time"] = int(t1 - t0)
    all_runs.to_csv(os.path.join(os.path.dirname(__file__), os.pardir, "runs", "runs.csv"), index=False)


def one_run(preproc: str, algo: str, **kwargs):
    (train_X, train_y), (test_X, test_y) = get_data()
    if preproc:
        trafo = PP[preproc]
        train_X = trafo.fit_transform(train_X)
        test_X = trafo.transform(test_X)
    classifier = globals()[algo]()
    best_model = GridSearchCV(estimator=classifier, param_grid=HP[algo], scoring="roc_auc_ovo", cv=5, verbose=3)
    best_model.fit(train_X, train_y)
    prediction = best_model.predict_proba(test_X)
    score = roc_auc_score(test_y, prediction, average="macro", multi_class="ovo")
    return score


def get_data() -> tuple:
    with open(os.path.join(os.path.dirname(__file__), os.pardir, "data", "mnist.pkl"), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
