#!/usr/bin/env python
from typing import List, Union, Optional, Any
import numpy
from classifier.classifier import Classifier


class Predictor(Model):
    """Set up model from available device for prediction job to run.
    """

    def __init__(self, model, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        super().__init__(model, model_path, device):
        self.model = model
        self.model_path = model_path
        self.device = device

    def predict_proba(self, inputs: List[Any], *args, **kwargs) -> numpy.ndarray:
        "Predict probability of class label given inputs."
        # toDo, think how to do both text & regular sklearn, as numpy.ndarray
        pass

    def predict(self, inputs: List[Any], *args, **kwargs) -> Union[numpy.ndarray, List]:
        "Predict class label given inputs."
        # toDo: think for multi-label problem, shall we return a indicator matrix constructed by orderred class labels
        pass
