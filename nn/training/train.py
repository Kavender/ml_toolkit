from typing import Dict, List, Union, Any, Callable
import os
import logging
from math import exp

from datasets import Dataset
from transformers.trainer_callback import TrainerCallback
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_utils import is_main_process, get_last_checkpoint, EvalPrediction
from transformers import CONFIG_MAPPING, Trainer, TrainingArguments, AutoTokenizer, default_data_collator
from transformers.utils.logging import set_verbosity_info, enable_default_handler, enable_explicit_format

from nn.utils import set_seed


# TODO: Tokenizer can be loaded here given the args definition or by default
# Datacollector loaded outside
def train(model, 
		  args: TrainingArguments = None,
          tokenizer,
          data_collator: Optional[DataCollator] = default_data_collator
          dataset_dict: Dict[str, Dataset],
		  optimizer: Optional[torch.optim.Optimizer] = None, 
		  scheduler: Optional[torch.optim.LambdaLR] = None,
          compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
		  callbacks: Optional[List[TrainerCallback]] = None) -> Trainer:

    # Setup logging
	log_level = args.get_process_log_level()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )

	# Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        set_verbosity_info()
        enable_default_handler()
        enable_explicit_format()
    
    logger.info("Training/evaluation parameters %s", args)

    # Set seed before initializing model.
    set_seed(args)

	# Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_dict['train'] if args.do_train else None,
        eval_dataset=dataset_dict['eval'] if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.do_eval and not is_torch_tpu_available() else None,
        callbacks=callbacks, 
        optimizers=(optimizer, scheduler),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if args.do_eval and not is_torch_tpu_available() else None,
    )


	if args.do_train:
		checkpoint = _identify_checkpoint(args)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        try:
            perplexity = exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

	return trainer



	





