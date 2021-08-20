import yaml
from ..nets import zoo


def parse(path):
    """Parse a config file for running a model.

    Arguments
    ---------
    path : str
        Path to the YAML-formatted config file to parse.

    Returns
    -------
    config : dict
        A `dict` containing the information from the config file at `path`.

    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    if not config['train'] and not config['infer']:
        raise ValueError('"train", "infer", or both must be true.')
    if config['train'] and config['training_data_csv'] is None:
        raise ValueError('"training_data_csv" must be provided if training.')
    if config['infer'] and config['inference_data_csv'] is None:
        raise ValueError('"inference_data_csv" must be provided if "infer".')
    if config['training']['lr'] is not None:
        config['training']['lr'] = float(config['training']['lr'])

    # TODO: IMPLEMENT UPDATING VALUES BASED ON EMPTY ELEMENTS HERE!

    if config['validation_augmentation'] is not None \
            and config['inference_augmentation'] is None:
        config['inference_augmentation'] = config['validation_augmentation']

    return config
