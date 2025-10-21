"""
LiteLLM-based multimodal inference for Solaris.

This module provides vision-language model inference capabilities using LiteLLM,
enabling analysis of overhead imagery with natural language queries using
US-based models (Google Gemini) and NVIDIA models.
"""

import os
import json
import base64
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from warnings import warn
import gdal
from PIL import Image
import io

try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    warn("LiteLLM is not installed. Install it with: pip install litellm")

from ..raster.image import create_multiband_geotiff
from ..utils.core import get_data_paths


class MultimodalInferer(object):
    """
    Vision-Language Model Inferer for overhead imagery using LiteLLM.

    This class enables multimodal analysis of satellite and aerial imagery
    using vision-language models like Google Gemini and NVIDIA models through
    the LiteLLM unified interface.

    Supported Models:
        - Google Gemini: gemini-1.5-pro, gemini-1.5-flash, gemini-pro-vision
        - NVIDIA models via NVIDIA API catalog
        - Any LiteLLM-compatible vision model

    Attributes:
        config (dict): Configuration dictionary containing model and inference parameters.
        model_name (str): Name of the vision-language model to use.
        output_dir (str): Directory to save inference results.
        batch_size (int): Batch size for inference.
        api_key (str): API key for the model provider.

    Example:
        >>> config = {
        ...     'model_name': 'gemini/gemini-1.5-flash',
        ...     'api_key': 'your-google-api-key',  # or set GEMINI_API_KEY env var
        ...     'litellm_inference': {
        ...         'output_dir': './litellm_outputs',
        ...         'prompt_template': 'Describe this satellite image in detail.',
        ...         'temperature': 0.7,
        ...         'max_tokens': 512
        ...     },
        ...     'inference_data_csv': 'images.csv'
        ... }
        >>> inferer = MultimodalInferer(config)
        >>> results = inferer()
    """

    def __init__(self, config: Dict):
        """
        Initialize the MultimodalInferer.

        Args:
            config: Configuration dictionary with keys:
                - model_name: Model identifier (e.g., 'gemini/gemini-1.5-flash')
                - api_key: API key for the provider (optional if set as env var)
                - litellm_inference: Dict containing:
                    - output_dir: Output directory path
                    - prompt_template: Text prompt for the model (optional)
                    - temperature: Sampling temperature (default: 0.7)
                    - max_tokens: Maximum tokens to generate (default: 512)
                    - top_p: Top-p sampling (default: 0.9)
                - batch_size: Batch size for inference (default: 1)
                - inference_data_csv: Path to CSV with image paths

        Raises:
            ImportError: If LiteLLM is not installed.
            ValueError: If required config parameters are missing.
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not installed. Install it with: pip install litellm\n"
                "For Google Gemini, also set GEMINI_API_KEY environment variable."
            )

        self.config = config
        self.model_name = config.get('model_name')
        if not self.model_name:
            raise ValueError("'model_name' must be specified in config")

        # Set API key if provided
        api_key = config.get('api_key')
        if api_key:
            # Set appropriate env var based on model provider
            if 'gemini' in self.model_name.lower():
                os.environ['GEMINI_API_KEY'] = api_key
            elif 'nvidia' in self.model_name.lower():
                os.environ['NVIDIA_API_KEY'] = api_key

        self.litellm_config = config.get('litellm_inference', {})
        if not self.litellm_config:
            raise ValueError("'litellm_inference' section must be present in config")

        # Output directory
        self.output_dir = self.litellm_config.get('output_dir', './litellm_outputs')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Prompt configuration
        self.prompt_template = self.litellm_config.get(
            'prompt_template',
            'Describe this overhead satellite or aerial image. '
            'Identify key features, land use, structures, and any notable patterns.'
        )

        # Sampling parameters
        self.temperature = self.litellm_config.get('temperature', 0.7)
        self.max_tokens = self.litellm_config.get('max_tokens', 512)
        self.top_p = self.litellm_config.get('top_p', 0.9)

        self.batch_size = config.get('batch_size', 1)

        # LiteLLM configuration
        litellm.set_verbose = config.get('verbose', False)

        print(f"Initialized LiteLLM with model: {self.model_name}")

    def _load_image(self, image_path: str) -> Tuple[Image.Image, Optional[str], Optional[tuple]]:
        """
        Load an image and extract geospatial metadata if available.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (PIL Image, projection, geotransform)
        """
        # Try to load geospatial metadata
        proj = None
        gt = None
        try:
            ds = gdal.Open(image_path)
            if ds is not None:
                proj = ds.GetProjection()
                gt = ds.GetGeoTransform()
                # Convert GDAL dataset to PIL Image
                array = ds.ReadAsArray()
                if len(array.shape) == 3:
                    # Transpose from (C, H, W) to (H, W, C)
                    array = np.transpose(array, (1, 2, 0))
                # Normalize to 0-255 if needed
                if array.dtype != np.uint8:
                    array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
                img = Image.fromarray(array)
        except Exception as e:
            warn(f"Could not load geospatial metadata from {image_path}: {e}")
            # Fall back to PIL
            img = Image.open(image_path)

        if img is None:
            img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img, proj, gt

    def _image_to_base64(self, img: Image.Image) -> str:
        """
        Convert PIL Image to base64 string for API calls.

        Args:
            img: PIL Image object

        Returns:
            Base64-encoded image string
        """
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def _format_prompt(self, image_path: str, custom_prompt: Optional[str] = None) -> str:
        """
        Format the prompt for the model.

        Args:
            image_path: Path to the image file.
            custom_prompt: Optional custom prompt to override template.

        Returns:
            Formatted prompt string.
        """
        if custom_prompt:
            return custom_prompt

        return self.prompt_template

    def _call_litellm(self, prompt: str, image_base64: str) -> str:
        """
        Call LiteLLM API with image and prompt.

        Args:
            prompt: Text prompt
            image_base64: Base64-encoded image

        Returns:
            Model response text
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": image_base64
                    }
                ]
            }
        ]

        response = completion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p
        )

        return response.choices[0].message.content

    def __call__(self, infer_df=None, prompts: Optional[List[str]] = None):
        """
        Run LiteLLM multimodal inference on a set of images.

        Args:
            infer_df: DataFrame with 'image' column containing image paths, or
                     path to CSV file. If None, uses config['inference_data_csv'].
            prompts: Optional list of custom prompts for each image. If provided,
                    must have same length as number of images.

        Returns:
            List of dictionaries containing inference results with keys:
                - image_path: Path to the input image
                - prompt: The prompt used
                - response: Model's text response
                - metadata: Additional metadata (projection, geotransform if available)
        """
        if infer_df is None:
            infer_df = get_infer_df(self.config)

        image_paths = infer_df['image'].tolist()

        if prompts is not None and len(prompts) != len(image_paths):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match number of "
                f"images ({len(image_paths)})"
            )

        results = []

        # Process images (note: batch processing depends on provider support)
        for idx, img_path in enumerate(image_paths):
            print(f"Processing image {idx + 1}/{len(image_paths)}: {img_path}")

            try:
                # Load image and metadata
                img, proj, gt = self._load_image(img_path)

                # Convert to base64
                img_base64 = self._image_to_base64(img)

                # Get prompt
                if prompts is not None:
                    prompt = prompts[idx]
                else:
                    prompt = self._format_prompt(img_path)

                # Call LiteLLM API
                response_text = self._call_litellm(prompt, img_base64)

                # Store result
                result = {
                    'image_path': img_path,
                    'prompt': prompt,
                    'response': response_text,
                    'metadata': {
                        'projection': proj,
                        'geotransform': gt
                    }
                }
                results.append(result)

                # Save individual result as JSON
                output_filename = os.path.splitext(os.path.basename(img_path))[0] + '_litellm.json'
                output_path = os.path.join(self.output_dir, output_filename)
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)

                print(f"Response: {result['response'][:200]}...")  # Print first 200 chars

            except Exception as e:
                warn(f"Error processing {img_path}: {e}")
                # Store error result
                result = {
                    'image_path': img_path,
                    'prompt': prompt if prompts else self.prompt_template,
                    'response': f"ERROR: {str(e)}",
                    'metadata': {'error': str(e)}
                }
                results.append(result)

        # Save all results
        all_results_path = os.path.join(self.output_dir, 'all_results.json')
        with open(all_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nInference complete! Results saved to: {self.output_dir}")
        print(f"Total images processed: {len(results)}")

        return results


def get_infer_df(config):
    """
    Get the inference dataframe based on the contents of config.

    Args:
        config: The loaded configuration dict for LiteLLM inference.

    Returns:
        DataFrame containing at least one column: 'image' with paths to image files.
    """
    infer_csv = config.get('inference_data_csv')
    if not infer_csv:
        raise ValueError("'inference_data_csv' must be specified in config")

    infer_df = get_data_paths(infer_csv, infer=True)
    return infer_df


# Keep backward compatibility with vllm_infer naming
VLLMInferer = MultimodalInferer
