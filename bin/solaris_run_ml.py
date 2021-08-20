import argparse
import os
import pandas as pd
from ..utils.config import parse
from ..nets.train import Trainer
from ..nets.infer import Inferer


def main():

    parser = argparse.ArgumentParser(
        description='Run a Solaris ML pipeline based on a config YAML',
        argument_default=None)

    parser.add_argument('--config', '-c', type=str, required=True,
                        help="Full path to a YAML-formatted config file "
                        "specifying parameters for model training and/or "
                        "inference.")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError('The configuration file cannot be found at the path '
                         'specified.')

    config = parse(args.config)

    if config['train']:
        trainer = Trainer(config)
        trainer.train()
    if config['infer']:
        inferer = Inferer(config)
        inf_df = pd.read_csv(config['inference_data_csv'])
        inferer(inf_df)


if __name__ == '__main__':
    main()
