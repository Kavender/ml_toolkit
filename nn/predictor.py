#!/usr/bin/env python
from typing import List, Union, Optional
import numpy
from classifier.classifier import Classifier


class Predictor(Model):
    """Set up model from available device for prediction job to run.
    """

    def __init__(self, model, model_path:Optional[str]) -> None:
        self.model = model

    def predict_proba(self, inputs: List[Any], *args, **kwargs) -> numpy.ndarray:
        "Predict probability of class label given inputs."
        # toDo, think how to do both text & regular sklearn, as numpy.ndarray
        pass

    def predict(self, inputs: List[Any], *args, **kwargs) -> Union[numpy.ndarray, List]:
        "Predict class label given inputs."
        # toDo: think for multi-label problem, shall we return a indicator matrix constructed by orderred class labels
        pass
