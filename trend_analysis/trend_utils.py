from typing import Iterable, Tuple
import numpy as np
import itertools

def pairwise(iterable: Iterable):
	"""s -> (s0,s1), (s1,s2), (s2, s3), ..."""
	a, b = itertools.tee(iterable)
	next(b, None)
	return zip(a, b)


def consecutive(data, stepsize=1):
	"""Split the list of ordered data if difference between two neighbors is greater than stepsize."""
	return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


###################
def split_sequence(sequence:Iterable, n_steps: int) -> Tuple[np.array, np.array]:
	"split a univariate sequence into samples"
	X = []
	y = []
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
