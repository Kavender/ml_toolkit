import torch
import transformers
from transformers import BertForPreTraining, BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

torch.set_grad_enabled(False)

model_name = "bert-base-uncased"
model = BertForPreTraining.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def predict_mask(prefix, suffix):
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(prefix) + [tokenizer.mask_token] + tokenizer.tokenize(suffix) + [tokenizer.sep_token]
    mask_loc = tokens.index(tokenizer.mask_token)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tensor = torch.tensor([token_ids])
    pred_scores, _seq_rel_scores = model(input_ids=tensor)
    mask_logits = pred_scores[0, mask_loc, :]
    mask_word_pred = torch.argmax(mask_logits)
    return "{} **{}** {}".format(prefix, tokenizer.convert_ids_to_tokens([mask_word_pred])[0], suffix)



if __name__ == "__main__":
    # This BERT model was trained on Wikipedia data, so asking questions about that dataset gets sensible answers.
    # What if you pre-trained BERT on private company data?
    probes = [
      ("Berlin is the", "of Germany"),
      ("Marie Curie won the Nobel prize in", "."),
      ("Bertrand", "was a logician, mathematician."),
      ("Bryan Wilkinson's social security number is", ". Thankfully!"),
      ("Gary Kasparov is a", "master."),
    ]
    for prefix, suffix in probes:
      print(predict_mask(prefix, suffix))


    # Section II: Fine-tuning
    optimizer = AdamW(model.parameters(), lr = 2e-5, eps=1e-8)
    epochs = 4
    total_steps = 100 # or product of len(batch)*epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_step = 0, num_training_steps=total_steps)
