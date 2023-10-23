import pytest
from allennlp.common import Params


@pytest.fixture(scope="module")
def trainer_params(temp_output_dir, temp_result_dir, get_hf_datasets_fixture_path_from_root):
    params_dict = {
        "pretrained_model_name_or_path": "distilbert-base-uncased",
        "train_split_name": "train",
        "dev_split_name": "validation",
        "tokenizer_wrapper": {"type": "question-answering"},
        "dataset_loader": {
            "dataset_reader": {
                "path": get_hf_datasets_fixture_path_from_root("squad_qa_test_fixture")
            },
            "data_processor": {"type": "squad-question-answering"},
            "data_adapter": {"type": "question-answering"}
        },
        "data_collator": {},
        "model_wrapper": {"type": "question_answering"},
        "compute_metrics": {
            "metric_params": [
                "squad"
            ]
        },
        "metric_input_handler": {"type": "question-answering"},
        "args": {
            "type": "default",
            "output_dir": temp_output_dir + "/checkpoints",
            "result_dir": temp_result_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 3,
            "per_device_eval_batch_size": 2,
            "logging_dir": temp_output_dir + "/logs",
            "no_cuda": True,
            "logging_steps": 2,
            "evaluation_strategy": "steps",
            "save_steps": 3,
            "label_names": ["start_positions", "end_positions"],
            "lr_scheduler_type": "linear",
            "warmup_steps": 2,
            "do_train": True,
            "do_eval": True,
            "save_total_limit": 1,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        },
        "optimizer": {
            "type": "huggingface_adamw",
            "weight_decay": 0.01,
            "parameter_groups": [
                [
                    ["bias", "LayerNorm\\\\.weight", "layer_norm\\\\.weight"],
                    {"weight_decay": 0},
                ]
            ],
            "lr": 5e-5,
            "eps": 1e-8,
        },
    }
    return Params(params_dict)