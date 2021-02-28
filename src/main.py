import os
import pickle

from bayesian_decision_tree.classification import PerpendicularClassificationTree, HyperplaneClassificationTree
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.hyper_params import HP
from src.preprocessors import PP


# __file__ = r"/home/oliver/PycharmProjects/bayesian_decision_tree_comparison/src/main.py"
def main():
    all_runs = pd.read_csv(os.path.join(os.path.dirname(__file__), os.pardir, "runs", "runs.csv"))
    for task in all_runs.itertuples():
        if np.isnan(task.roc):
            score, refit_time = one_run(task.algo)
            all_runs.loc[task.Index, "roc"] = score
            all_runs.loc[task.Index, "time"] = refit_time
            all_runs.to_csv(os.path.join(os.path.dirname(__file__), os.pardir, "runs", "runs.csv"), index=False)
    copy_info(all_runs)


def one_run(algo: str, **kwargs):
    (train_X, train_y), (test_X, test_y) = get_data()
    classifier = globals()[algo]()
    best_model = GridSearchCV(estimator=classifier, param_grid=HP[algo], scoring="roc_auc_ovo", cv=5, verbose=3)
    best_model.fit(train_X, train_y)
    prediction = best_model.predict_proba(test_X)
    score = roc_auc_score(test_y, prediction, average="macro", multi_class="ovo")
    return score, best_model.refit_time_


def get_data() -> tuple:
    with open(os.path.join(os.path.dirname(__file__), os.pardir, "data", "mnist.pkl"), "rb") as f:
        return pickle.load(f)


def copy_info(all_runs: pd.DataFrame):
    """Writes the info to the csv generated by the notebook"""
    runs = all_runs.set_index("algo")
    time_file = os.path.join(os.path.dirname(__file__), os.pardir, "data", "times.csv")
    results_file = os.path.join(os.path.dirname(__file__), os.pardir, "data", "results.csv")
    times = pd.read_csv(time_file)
    results = pd.read_csv(results_file)
    times.loc["mnist"] = {
        "problem": "mnist",
        "dt": runs.loc["DecisionTreeClassifier", "time"],
        "bdt": runs.loc["PerpendicularClassificationTree", "time"],
        "rf": runs.loc["RandomForestClassifier", "time"],
        "xgb": runs.loc["XGBClassifier", "time"],
        "lgb": runs.loc["LGBMClassifier", "time"],
    }
    results.loc["mnist"] = {
        "problem": "mnist",
        "dt": runs.loc["DecisionTreeClassifier", "roc"],
        "bdt": runs.loc["PerpendicularClassificationTree", "roc"],
        "rf": runs.loc["RandomForestClassifier", "roc"],
        "xgb": runs.loc["XGBClassifier", "roc"],
        "lgb": runs.loc["LGBMClassifier", "roc"],
    }
    times.to_csv(time_file, index=False)
    results.to_csv(results_file, index=False)


if __name__ == "__main__":
    main()
