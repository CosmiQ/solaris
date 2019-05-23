import os
import skimage
import torch
from .model_io import get_model
from .transform import process_aug_dict
from .datagen import InferenceTiler
from ..raster.image import stitch_images
from ..utils.core import get_data_paths


class Inferer(object):
    """Object for training `solaris` models using PyTorch or Keras."""

    def __init__(self, config):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.framework = self.config['nn_framework']
        self.model_name = self.config['model_name']
        # check if the model was trained as part of the same pipeline; if so,
        # use the output from that. If not, use the pre-trained model directly.
        if self.config['train']:
            self.model_path = self.config['training']['model_dest_path']
        else:
            self.model_path = self.config['model_path']
        self.model = get_model(self.model_name, self.framework,
                               self.model_path)
        self.window_step_x = self.config['inference'].get('window_step_size_x')
        self.window_step_y = self.config['inference'].get('window_step_size_y')
        if self.window_step_x is None:
            self.window_step_x = self.config['data_specs']['width']
        if self.window_step_y is None:
            self.window_step_y = self.config['data_specs']['height']
        self.stitching_method = self.config['inference'].get(
            'stitching_method', 'average')
        self.output_dir = self.config['inference']['output_dir']

    def __call__(self, infer_df):
        """Run inference.

        Arguments
        ---------
        infer_df : :class:`pandas.DataFrame` or `str`
            A :class:`pandas.DataFrame` with a column, ``'image'``, specifying
            paths to images for inference. Alternatively, `infer_df` can be a
            path to a CSV file containing the same information.

        """
        inf_tiler = InferenceTiler(
            self.framework,
            width=self.config['data_specs']['width'],
            height=self.config['data_specs']['height'],
            x_step=self.window_step_x,
            y_step=self.window_step_y,
            augmentations=process_aug_dict(
                self.config['inference']['inference_augmentation'])
            )
        for im_path in infer_df['image']:
            inf_input, idx_refs, (
                src_im_height, src_im_width) = inf_tiler(im_path)

            if self.framework == 'keras':
                subarr_preds = self.model.predict(inf_input,
                                                  batch_size=self.batch_size)

            elif self.framework in ['torch', 'pytorch']:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    self.model = self.model.cuda()
                else:
                    device = torch.device('cpu')
                inf_input = torch.from_numpy(inf_input).to(device)
                subarr_preds = self.model(inf_input)

            stitched_result = stitch_images(subarr_preds.data.numpy(),
                                            idx_refs=idx_refs,
                                            out_width=src_im_width,
                                            out_height=src_im_height,
                                            method=self.stitching_method)
            skimage.io.imsave(os.path.join(self.output_dir,
                                           os.path.split(im_path)[1]),
                              stitched_result)


def get_infer_df(config):
    """Get the inference df based on the contents of ``config`` .

    This function uses the logic described in the documentation for the config
    file to determine where to find images to be used for inference.
    See the docs and the comments in solaris/data/config_skeleton.yml for
    details.

    Arguments
    ---------
    config : dict
        The loaded configuration dict for model training and/or inference.

    Returns
    -------
    infer_df : :class:`dict`
        :class:`dict` containing at least one column: ``'image'`` . The values
        in this column correspond to the path to filenames to perform inference
        on.
    """

    infer_df = get_data_paths(config['inference_data_csv'], infer=True)
    return infer_df
