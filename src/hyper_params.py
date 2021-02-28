import numpy as np

from bayesian_decision_tree.hyperplane_optimization import SimulatedAnnealingOptimizer


HP = {
    "RandomForestClassifier": {
        "n_estimators": [100, 500],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10],
        "class_weight": ["balanced"],
        "n_jobs": [-1]
    },
    "XGBClassifier": {
        "n_estimators": [100, 250],
        "max_depth": [2, 3],
        "n_jobs": [-1],
        "use_label_encoder": [False],
    },
    "PerpendicularClassificationTree":  {
        "partition_prior": [0.9, 0.99, 0.8],
        "prior": [np.ones(10)],
        "prune": [True, False],
    },
    "LGBMClassifier": {
        "boosting_type": ["gbdt", "dart"],
        "max_depth": [-1, 3],
    },
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
    }
}
