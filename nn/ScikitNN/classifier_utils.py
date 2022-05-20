import os
import argparse
from typing import Optional


def str2bool(v):
    """Utility function for parsing boolean in argparse
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    :param v: value of the argument
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_metadata_file_path(checkpoint_prefix: str) -> str:
    """Get the checkpoint file path for meta data"""
    return checkpoint_prefix + '_metadata.pkl'


def get_file_from_checkpoint(checkpoint_prefix: str) -> str:
    "Get the last checkpoint file saved in directory."
    dirname, prefix = os.path.split(checkpoint_prefix)
    checkpoint_files = [fname for fname in os.listdir(dirname) if fname.startswith(prefix)]
                        # and fname.endswith(os.path.extsep + "params")
    last_checkpoint_filename = max(checkpoint_files) ##wait, doesn't the checkpoint to be str?
    last_checkpoint_path = os.path.join(dirname, last_checkpoint_filename)
    return last_checkpoint_path

## we have to generalize it either as the how gluonnlp.model.get_model
# def get_bert_model(model_name, cased, ctx, dropout_prob):
#     """Get pre-trained BERT model."""
#
#     return nlp.model.get_model(
#         name=model_name,
#         dataset_name=bert_dataset_name,
#         pretrained=True,
#         ctx=ctx,
#         use_pooler=False,
#         use_decoder=False,
#         use_classifier=False,
#         dropout=dropout_prob,
#         embed_dropout=dropout_prob)
