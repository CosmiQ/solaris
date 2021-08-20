import os
import torch
import gdal
import numpy as np
from warnings import warn
from .model_io import get_model
from .transform import process_aug_dict
from .datagen import InferenceTiler
from ..raster.image import stitch_images, create_multiband_geotiff
from ..utils.core import get_data_paths


class Inferer(object):
    """Object for training `solaris` models using PyTorch or Keras."""

    def __init__(self, config, custom_model_dict=None):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.framework = self.config['nn_framework']
        self.model_name = self.config['model_name']
        # check if the model was trained as part of the same pipeline; if so,
        # use the output from that. If not, use the pre-trained model directly.
        if self.config['train']:
            warn('Because the configuration specifies both training and '
                 'inference, solaris is switching the model weights path '
                 'to the training output path.')
            self.model_path = self.config['training']['model_dest_path']
            if custom_model_dict is not None:
                custom_model_dict['weight_path'] = self.config[
                    'training']['model_dest_path']
        else:
            self.model_path = self.config.get('model_path', None)
        self.model = get_model(self.model_name, self.framework,
                               self.model_path, pretrained=True,
                               custom_model_dict=custom_model_dict)
        self.window_step_x = self.config['inference'].get('window_step_size_x',
                                                          None)
        self.window_step_y = self.config['inference'].get('window_step_size_y',
                                                          None)
        if self.window_step_x is None:
            self.window_step_x = self.config['data_specs']['width']
        if self.window_step_y is None:
            self.window_step_y = self.config['data_specs']['height']
        self.stitching_method = self.config['inference'].get(
            'stitching_method', 'average')
        self.output_dir = self.config['inference']['output_dir']
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def __call__(self, infer_df=None):
        """Run inference.
        Arguments
        ---------
        infer_df : :class:`pandas.DataFrame` or `str`
            A :class:`pandas.DataFrame` with a column, ``'image'``, specifying
            paths to images for inference. Alternatively, `infer_df` can be a
            path to a CSV file containing the same information.  Defaults to
            ``None``, in which case the file path specified in the Inferer's
            configuration dict is used.
        """

        if infer_df is None:
            infer_df = get_infer_df(self.config)

        inf_tiler = InferenceTiler(
            self.framework,
            width=self.config['data_specs']['width'],
            height=self.config['data_specs']['height'],
            x_step=self.window_step_x,
            y_step=self.window_step_y,
            augmentations=process_aug_dict(
                self.config['inference_augmentation']))
        for idx, im_path in enumerate(infer_df['image']):
            temp_im = gdal.Open(im_path)
            proj = temp_im.GetProjection()
            gt = temp_im.GetGeoTransform()
            inf_input, idx_refs, (
                src_im_height, src_im_width) = inf_tiler(im_path)

            if self.framework == 'keras':
                subarr_preds = self.model.predict(inf_input,
                                                  batch_size=self.batch_size)

            elif self.framework in ['torch', 'pytorch']:
                with torch.no_grad():
                    self.model.eval()
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    self.model = self.model.cuda()
                else:
                    device = torch.device('cpu')
                inf_input = torch.from_numpy(inf_input).float().to(device)
                # add additional input data, if applicable
                if self.config['data_specs'].get('additional_inputs',
                                                 None) is not None:
                    inf_input = [inf_input]
                    for i in self.config['data_specs']['additional_inputs']:
                        inf_input.append(
                            infer_df[i].iloc[idx].to(device))

                subarr_preds = self.model(inf_input)
                subarr_preds = subarr_preds.cpu().data.numpy()
            stitched_result = stitch_images(subarr_preds,
                                            idx_refs=idx_refs,
                                            out_width=src_im_width,
                                            out_height=src_im_height,
                                            method=self.stitching_method)
            stitched_result = np.swapaxes(stitched_result, 1, 0)
            stitched_result = np.swapaxes(stitched_result, 2, 0)
            create_multiband_geotiff(stitched_result,
                                     os.path.join(self.output_dir,
                                                  os.path.split(im_path)[1]),
                                     proj=proj, geo=gt, nodata=np.nan,
                                     out_format=gdal.GDT_Float32)


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
