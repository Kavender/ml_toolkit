#!/usr/bin/env python
from typing import Dict, Union, Optional
from abc import ABCMeta, abstractmethod
from pathlib import Path
from argparse import Namespace
import numpy
import yaml
import torch
import joblib


class Learner(ABCMeta):
    """Abstract parent learner class for machine learning and neural net classifiers.
    """

    def __init__(self, config: Optional[dict, Namespace]):
        self.config = config
        self.model = None

    @abstractmethod
    def fit(self, X, y) -> None:
        return NotImplementedError

    @abstractmethod
    def predict(self, X) -> numpy.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X) -> numpy.ndarray:
        raise NotImplementedError

    def build(self):
        raise NotADirectoryError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError


class DummyLearner(Learner):
    """Semi-abstract learner class, provide commonly used supporting functions, such as load/save model or model config.
    Later, we want to add more functions, such as save and load from s3 (toDo)
    """

    def save_model(self, file_path: str, model_name: str) -> None:
        "Store model along with config into defined location."
        if not isinstance(file_path, Path):
            path = Path(file_path)
        path.mkdir(exist_ok=True)
        joblib.dump(self.model, Path(path, model_name + ".pkl"))
        self.save_model_config(file_path, model_name)

    def load_model(self, file_path: str, model_name: str):
        "Load model and config for prediction, training or inspection."
        self.model = joblib.load(Path(path, model_name + ".pkl"))
        self.config = self.load_model_config(Path(path, model_name + "_config.yml"))

    def save_model_config(self, file_path: str, model_name: str) -> None:
        "Store model params into yaml file for checkpoint or reproduce."
        yaml_path = Path(path, model_name + "_config.yml")
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(vars(self.config), yaml_file)

    def load_model_config(self, yaml_path: Optional[str], params: Optional[dict, Namespace]) -> Namespace:
        "Retrieve model params either from a yaml file or directly pass in value."
        if yaml_path:
            return self.get_params_from_yaml(yaml_path)
        if params:
            return self.convert_params_to_namespace(params)
        raise VauleError("Please specify either a yaml file or default params.")

    @staticmethod
    def get_params_from_yaml(yaml_path: str) -> Namespace:
        "Read parameters stored in yaml file"
        with open(yaml_path, "r", encoding="utf-8") as yaml_file:
            param_values = yaml.safe_load(yaml_file)
            return Namespace(**param_values)

    @staticmethod
    def convert_params_to_namespace(params: Optional[dict, Namespace]) -> Namespace:
        "Make non-callable params to namespace."
        if isinstance(params, Namespace):
            return params
        for argname, arg_value in vars(params).items():
            if callable(arg_value):
                del params[argname]
        return Namespace(**params)
