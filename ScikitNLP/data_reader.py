import ast
from ..utils.types import TaggedTokens, PredictedToken


###maybe btter to have a dataset module and reader function per model ?
NULL_TAG = 'X'

def load_json_examples(file_path):
    "Read json from local file."
    return list(map(ast.literal_eval, open(file_path)))

# toDO: need to refactor for our usecase
def bio_bioes(tokens):
    """Convert a list of TaggedTokens in BIO(2) scheme to BIOES scheme.
    Parameters
    ----------
    tokens: List[TaggedToken]
        A list of tokens in BIO(2) scheme
    Returns
    -------
    List[TaggedToken]:
        A list of tokens in BIOES scheme
    """
    ret = []
    for index, token in enumerate(tokens):
        if token.tag == 'O':
            ret.append(token)
        elif token.tag.startswith('B'):
            # if a B-tag is continued by other tokens with the same entity,
            # then it is still a B-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith('I'):
                ret.append(token)
            else:
                ret.append(TaggedToken(text=token.text, tag='S' + token.tag[1:]))
        elif token.tag.startswith('I'):
            # if an I-tag is continued by other tokens with the same entity,
            # then it is still an I-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith('I'):
                ret.append(token)
            else:
                ret.append(TaggedToken(text=token.text, tag='E' + token.tag[1:]))
    return ret


def convert_arrays_to_text(text_vocab, tag_vocab,
                           np_text_ids, np_true_tags, np_pred_tags, np_valid_length):
    """Convert numpy array data into text
    Parameters
    ----------
    np_text_ids: token text ids (batch_size, seq_len)
    np_true_tags: tag_ids (batch_size, seq_len)
    np_pred_tags: tag_ids (batch_size, seq_len)
    np.array: valid_length (batch_size,) the number of tokens until [SEP] token
    Returns
    -------
    List[List[PredictedToken]]:
    """
    predictions = []
    for sample_index in range(np_valid_length.shape[0]):
        sample_len = np_valid_length[sample_index]
        entries = []
        for i in range(1, sample_len - 1):
            token_text = text_vocab.idx_to_token[np_text_ids[sample_index, i]]
            true_tag = tag_vocab.idx_to_token[int(np_true_tags[sample_index, i])]
            pred_tag = tag_vocab.idx_to_token[int(np_pred_tags[sample_index, i])]
            # we don't need to predict on NULL tags
            if true_tag == NULL_TAG:
                last_entry = entries[-1]
                entries[-1] = PredictedToken(text=last_entry.text + token_text,
                                             true_tag=last_entry.true_tag,
                                             pred_tag=last_entry.pred_tag)
            else:
                entries.append(PredictedToken(text=token_text,
                                              true_tag=true_tag, pred_tag=pred_tag))

        predictions.append(entries)
    return predictions
