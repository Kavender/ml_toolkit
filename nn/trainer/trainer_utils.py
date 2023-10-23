from typing import Union, Any
import os
import time
import secrets
import logging
import tempfile
from pathlib import Path
from datasets import load_dataset
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from transformers.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


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


def load_config(args: Any, config_kwargs: dict) -> Any:
    """
    Loads a configuration based on given arguments.

    Parameters:
    - args: Object that contains configuration attributes.
    - config_kwargs (dict): Additional keyword arguments for configuration.

    Returns:
    - A configuration object.

    Raises:
    - KeyError: If `args.model_type` is not found in `CONFIG_MAPPING`.
    """
    
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING_NAMES.get(args.model_type)
        if config is None:
            raise KeyError(f"'{args.model_type}' is not a valid model type in CONFIG_MAPPING.")
        config = config()
        logger.warning("You are instantiating a new config instance from scratch.")
        
        if hasattr(args, 'config_overrides') and args.config_overrides is not None:
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


def identify_checkpoint(args: TrainingArguments) -> Union[str, None]:
	"""
	Identifies the checkpoint to resume from.

	Parameters:
	- args (TrainingArguments): Object containing the training arguments.

	Returns:
	- str: Path of the checkpoint to resume from, or None if not found.
	"""
	if args.resume_from_checkpoint is not None:
		checkpoint = args.resume_from_checkpoint
	else:
		checkpoint = _detect_last_checkpoint(args)
	return checkpoint


def _detect_last_checkpoint(args: TrainingArguments) -> str:
    """
    Detects the last checkpoint from the output directory if exists.

    Parameters:
    - args (TrainingArguments): Object containing the training arguments.

    Returns:
    - str: Path of the last checkpoint, or None if not found.

    Raises:
    - ValueError: If the output directory exists, is not empty, and overwrite is not specified.
    """
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


def _create_serialization_dirs(args: TrainingArguments) -> str:
    """
    Creates or verifies the serialization directory based on given arguments.

    Parameters:
    - args (TrainingArguments): Object containing the training arguments.

    Returns:
    - str: Path of the serialization directory.

    Raises:
    - ValueError: If the specified directory exists and is not empty.
    """
    serialization_dir = getattr(args, "output_dir", None)
    
    if serialization_dir is None:
        serialization_dir = tempfile.mkdtemp()

    if not os.path.isdir(serialization_dir):
        os.makedirs(serialization_dir)
    elif len(os.listdir(serialization_dir)) > 0:
        raise ValueError(
            f"Serialization directory: `{serialization_dir}` not empty. Provide an "
            f"empty or non-existent directory."
        )
    
    return serialization_dir


def generate_model_filename(model_name):
    token = secrets.token_urlsafe(6)
    filename = f"{model_name}-{time.strftime('%Y%m%d')}-{token}"

    while os.path.exists(filename):
         generate_model_filename(model_name)
    return filename


def create_default_loggers(model_name: str):
    logger_name = generate_model_filename(model_name)
    
    # Create WandB and TensorBoard loggers
    wandb_logger = WandbLogger(name=logger_name, log_model='all')
    tensorboard_logger = TensorBoardLogger('lightning_logs', name=logger_name)
    
    return [wandb_logger, tensorboard_logger]
