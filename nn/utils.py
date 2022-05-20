from typing import Dict, Union, Any
import random
import torch
import numpy as np
from argparse import Namespace


def set_seed(args: Union[Dict[str, Any], Namespace, TrainingAurguments]):
	if isinstance(args, Namespace):
		args = vars(args)
	elif isinstance(args, TrainingAurguments):
		args = args.to_dict()

	seed = args.get("seed": 123)
	random.seed(seed)
	np.random.seed(seed)
	torch.manaul_seed(seed):
	# set all GPUs with same seed
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


