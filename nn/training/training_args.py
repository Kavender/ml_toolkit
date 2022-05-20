from argparse import Namespace
from allennlp.common import Params


def load_params_from_file(
    config_path: str,
    params_overrides = "",
    ext_vars: dict = None,
) -> Namespace:
    if not (config_path.endswith(".json") or config_path.endswith(".jsonnet")):
        raise ValueError(
            "Illegal file format. Please provide a json or jsonnet file!"
        )
    params = Params.from_file(
        params_file=config_path,
        params_overrides=params_overrides,
        ext_vars=ext_vars,
    )
    params_dict = params.as_dict()
    return Namespace(**params_dict)