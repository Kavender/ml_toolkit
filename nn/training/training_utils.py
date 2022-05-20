from typing import Dict
from pathlib import Path
from datasets import DatasetDict, load_dataset


def _get_file_extension(file_path, extension_mapping={'txt': 'text'}):
	file_ext = Path(file_path).suffix[1:]
	return extension_mapping.get(file_ext, file_ext)


def prepare_datadict(data_args):
    data_files = {}
    dataset_args = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
    	data_files["test"] = data_args.test_file
    
    extension = _get_file_extension(data_args.train_file) if data_args.train_file else _get_file_extension(data_args.validation_file)
  
    if extension == 'text':
    	dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
    loaded_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in loaded_datasets.keys():
        loaded_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            **dataset_args,
        )
        loaded_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            **dataset_args,
        )
     return loaded_datasets


def preprocess_logits_for_eval(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def validate_required_fileds(data, required_cols):
	if isinstance(required_cols, str):
		required_cols = [required_cols]
    for col_name in required_cols:
    	if col_name not in df.column_names:
		    raise ValueError(f"column- {col_name} not found in dataset. "
		        "Make sure to set column {column} to the correct audio column - one of {', '.join(data.column_names)}."
		    )
		    return False
    return True


def load_config(args, config_kwargs):
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if args.config_overrides is not None:
            logger.info(f"Overriding config: {args.config_overrides}")
            config.update_from_string(args.config_overrides)
            logger.info(f"New config: {config}")
        return config


def load_tokenizer(args, tokenizer_kwargs):
    if args.tokenizer_name:
        return AutoTokenizer.from_pretrained(args.tokenizer_name, **tokenizer_kwargs)
    elif args.model_name_or_path:
        return AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("Initalizing a new tokenizer is not supported! Make sure you load existing using --tokenizer_name")


def identify_checkpoint(args):
	checkpoint = None
	last_checkpoint = _detect_last_checkpoint(args)
	if args.resume_from_checkpoint is not None:
		checkpoint = args.resume_from_checkpoint
	elif last_checkpoint is not None:
		checkpoint = last_checkpoint
	return checkpoint


def _detect_last_checkpoint(args: TrainingAurguments):
	last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint



def _create_serialization_dirs(args: TrainingAurguments) -> List[str]:
    serialization_dir = args.get("output_dir", None)
    
    if serialization_dir is None:
        import tempfile
        serialization_dir = tempfile.mkdtemp()
    if not os.path.isdir(serialization_dir):
        os.makedirs(serialization_dir)
    elif len(os.listdir(serialization_dir)) > 0:
        raise ValueError(
            f"Serialization directory: `{serialization_dir}` not emtpy. Provide an "
            f"empty or non-existent directory."
        )
    return serialization_dir



