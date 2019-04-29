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
    if config['model_name'] not in zoo.models:
        raise ValueError('{} is not a valid model name.'.format(
            config['model_name']))
    if not config['train'] and not config['infer']:
        raise ValueError('"train", "infer", or both must be true.')
    if config['train'] and config['data']['train_im_src'] is None:
        raise ValueError('"train_im_src" must be provided if training.')
    if config['train'] and config['data']['train_label_src'] is None:
        raise ValueError('"train_label_src" must be provided if training.')
    if config['infer'] and config['data']['infer_im_src'] is None:
        raise ValueError('"infer_im_src" must be provided if "infer".')
    # TODO: IMPLEMENT UPDATING VALUES BASED ON EMPTY ELEMENTS HERE!

    return config
