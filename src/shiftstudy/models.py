"""Small tabular baselines and a paired single/ensemble MLP comparison."""

from copy import deepcopy
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .metrics import nll


def aligned_probabilities(model, x, n_classes):
    p = np.zeros((len(x), n_classes))
    p[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
    return p


def train_baselines(x, y, config, n_classes):
    models, diagnostics = {}, {}
    specifications = {
        "logistic": LogisticRegression(C=config["logistic_c"],
            max_iter=config["logistic_max_iter"], solver="lbfgs", random_state=0),
        "hist_boost": HistGradientBoostingClassifier(
            max_iter=config["boost_iterations"], max_leaf_nodes=config["boost_leaf_nodes"],
            learning_rate=config["boost_learning_rate"], early_stopping=False, random_state=0),
    }
    for name, model in specifications.items():
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x, y)
        models[name] = model
        diagnostics[name] = {"fit_seconds": time.perf_counter() - started,
                              "warnings": [str(w.message) for w in caught]}
    prior = np.bincount(y, minlength=n_classes) / len(y)
    return models, prior, diagnostics


def member_seed(outer_seed, member):
    return int(np.random.SeedSequence([outer_seed, member, 1741]).generate_state(1)[0])


def train_mlp(x, y, validation_x, validation_y, config, seed, n_classes):
    model = MLPClassifier(hidden_layer_sizes=(config["hidden_units"],),
        activation="relu", solver="adam", alpha=config["l2"],
        batch_size=min(config["batch_size"], len(x)),
        learning_rate_init=config["learning_rate"], shuffle=True,
        early_stopping=False, random_state=seed)
    best, best_nll, patience_nll = None, np.inf, np.inf
    wait = 0
    history = []
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(1, config["max_epochs"] + 1):
            model.partial_fit(x, y, classes=np.arange(n_classes))
            val_nll = nll(aligned_probabilities(model, validation_x, n_classes), validation_y)
            history.append({"epoch": epoch, "train_loss": float(model.loss_),
                            "validation_nll": val_nll})
            if val_nll < best_nll:
                best, best_nll, best_epoch = deepcopy(model), val_nll, epoch
            if val_nll < patience_nll - config["min_delta"]:
                patience_nll, wait = val_nll, 0
            else:
                wait += 1
            if wait >= config["patience"]:
                break
    diagnostics = {"seed": seed, "fit_seconds": time.perf_counter() - started,
                   "selected_epoch": best_epoch, "epochs_run": len(history),
                   "selected_validation_nll": best_nll,
                   "reached_epoch_budget": len(history) == config["max_epochs"],
                   "warnings": [str(w.message) for w in caught], "history": history}
    return best, diagnostics
