"""Closed configuration schema: fail on typos instead of changing the experiment."""

from pathlib import Path
import json
import math

TOP = {"preparation", "seeds", "ensemble_size", "threads", "alpha", "mlp", "baselines"}
MLP = {"hidden_units", "batch_size", "learning_rate", "l2", "max_epochs", "patience", "min_delta"}
BASELINES = {"logistic_c", "logistic_max_iter", "boost_iterations", "boost_leaf_nodes", "boost_learning_rate"}


def keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{where} must contain exactly {sorted(expected)}")


def positive_integer(value, where, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")


def number(value, where, *, allow_zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (not allow_zero and value == 0)):
        raise ValueError(f"{where} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def validate_config(config):
    keys(config, TOP, "configuration")
    keys(config["preparation"], {"seed", "sizes"}, "preparation")
    keys(config["preparation"]["sizes"], {"train", "dev_in", "eval_in", "eval_out"}, "sizes")
    positive_integer(config["preparation"]["seed"], "preparation.seed", 0)
    for name, value in config["preparation"]["sizes"].items():
        positive_integer(value, f"sizes.{name}", 3)
    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise ValueError("seeds must be a nonempty list")
    for value in config["seeds"]:
        positive_integer(value, "seed", 0)
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Duplicate experiment seeds")
    positive_integer(config["ensemble_size"], "ensemble_size", 2)
    positive_integer(config["threads"], "threads")
    number(config["alpha"], "alpha")
    if config["alpha"] >= 1:
        raise ValueError("alpha must be < 1")
    keys(config["mlp"], MLP, "mlp")
    keys(config["baselines"], BASELINES, "baselines")
    for name, value in config["mlp"].items():
        if name in {"hidden_units", "batch_size", "max_epochs", "patience"}:
            positive_integer(value, f"mlp.{name}")
        else:
            number(value, f"mlp.{name}", allow_zero=name in {"l2", "min_delta"})
    for name, value in config["baselines"].items():
        if name in {"logistic_max_iter", "boost_iterations", "boost_leaf_nodes"}:
            positive_integer(value, f"baselines.{name}", 2)
        else:
            number(value, f"baselines.{name}")
    return config


def load_config(path):
    def reject_constant(value):
        raise ValueError(f"Non-standard JSON number: {value}")
    return validate_config(json.loads(Path(path).read_text(), parse_constant=reject_constant))
