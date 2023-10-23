from pathlib import Path
import yaml
from argparse import Namespace
from allennlp.common import Params


TEST_DIR = "tests/data"
data_path = Path(TEST_DIR, "sequence_tagging.tsv")
config_path = Path(TEST_DIR, "config.json")
config_json = """{
        "model": {
                "type": "duplicate-test-tagger",
                "text_field_embedder": {
                        "token_embedders": {
                                "tokens": {
                                        "type": "embedding",
                                        "embedding_dim": 5
                                }
                        }
                },
                "encoder": {
                        "type": "lstm",
                        "input_size": 5,
                        "hidden_size": 7,
                        "num_layers": 2
                }
        },
        "dataset_reader": {"type": "sequence_tagging"},
        "train_data_path": "$$$",
        "validation_data_path": "$$$",
        "data_loader": {"batch_size": 2},
        "trainer": {
                "num_epochs": 2,
                "optimizer": "adam"
        }
    }""".replace(
    "$$$", str(data_path)
)

with open(config_path, "w") as config_file:
    config_file.write(config_json)


config_path = Path(TEST_DIR, 'config.yaml')
with open(config_path, 'w') as outfile:
    yaml.dump(config_json, outfile, default_flow_style=False)


with open(config_path) as fh:
    read_data = yaml.load(fh, Loader=yaml.FullLoader)




