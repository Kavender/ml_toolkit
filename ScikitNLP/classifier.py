#!/usr/bin/env python
from typing import Union, List, Dict, Tuple, Any
from argparse import Namespace
from pathlib import Path
from classifier.base import DummyLearner
from sklearn.base import ClassifierMixin
import torch ##maybe we don't need if nn has check for device func
import mxnet as mx
from mxnet.gluon import Block, nn

#so, actually, we need a sklearn classifier and a transformer classifier (built on MXnet instead of pytorch),
# could have two separate for predictor as well
# but our goal is to fit them all into sklearn-like pipeline (how possible, no idea)


class Classifier(DummyLearner, ClassifierMixin):
    def __init__(self, module, classes=None, train_split=CVSplit(5, stratified=True),
                *args,**kwargs):
        self.classes = classes



class TransformerClassifier(Block, nn):
    # https://github.com/allenai/allennlp/blob/master/allennlp/models/model.py
    # should also have a to_tensorboard() function if not implemented

    super(DummyLearner).__init__(self, *args, **kwargs):
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.is_fine_tuned = False

    def forward(self, **kwargs) -> Tuple:
        """Defined forward pass of the classifier, wraps around transformer's forward method.
        """

        return self.model.forward(**kwargs)


    def clean_up_gpu(self):
        "Free up memory in GPU by setting model to CPU"
        self.model.to("cpu")
        torch.cuda.empty_cache()
