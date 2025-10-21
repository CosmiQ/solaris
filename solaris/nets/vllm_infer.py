"""
vLLM-based multimodal inference for Solaris.

This module provides vision-language model inference capabilities using vLLM,
enabling analysis of overhead imagery with natural language queries.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from warnings import warn
import gdal
from PIL import Image

try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal.image import ImagePixelData
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    warn("vLLM is not installed. Install it with: pip install vllm")

from ..raster.image import create_multiband_geotiff
from ..utils.core import get_data_paths


class VLLMInferer(object):
    """
    Vision-Language Model Inferer for overhead imagery using vLLM.

    This class enables multimodal analysis of satellite and aerial imagery
    using vision-language models like LLaVA, Qwen-VL, and others supported by vLLM.

    Attributes:
        config (dict): Configuration dictionary containing model and inference parameters.
        model_name (str): Name/path of the vision-language model to use.
        model (LLM): vLLM model instance.
        sampling_params (SamplingParams): Parameters for text generation.
        output_dir (str): Directory to save inference results.
        batch_size (int): Batch size for inference.

    Example:
        >>> config = {
        ...     'model_name': 'llava-hf/llava-1.5-7b-hf',
        ...     'vllm_inference': {
        ...         'output_dir': './vllm_outputs',
        ...         'prompt_template': 'Describe this satellite image in detail.',
        ...         'temperature': 0.7,
        ...         'max_tokens': 512
        ...     },
        ...     'inference_data_csv': 'images.csv'
        ... }
        >>> inferer = VLLMInferer(config)
        >>> results = inferer()
    """

    def __init__(self, config: Dict):
        """
        Initialize the VLLMInferer.

        Args:
            config: Configuration dictionary with keys:
                - model_name: Model identifier (HuggingFace model or local path)
                - vllm_inference: Dict containing:
                    - output_dir: Output directory path
                    - prompt_template: Text prompt for the model (optional)
                    - temperature: Sampling temperature (default: 0.7)
                    - max_tokens: Maximum tokens to generate (default: 512)
                    - top_p: Top-p sampling (default: 0.9)
                    - top_k: Top-k sampling (default: -1, disabled)
                - batch_size: Batch size for inference (default: 1)
                - inference_data_csv: Path to CSV with image paths
                - tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
                - gpu_memory_utilization: GPU memory utilization (default: 0.9)

        Raises:
            ImportError: If vLLM is not installed.
            ValueError: If required config parameters are missing.
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm\n"
                "For vision models, you may need: pip install vllm[vision]"
            )

        self.config = config
        self.model_name = config.get('model_name')
        if not self.model_name:
            raise ValueError("'model_name' must be specified in config")

        self.vllm_config = config.get('vllm_inference', {})
        if not self.vllm_config:
            raise ValueError("'vllm_inference' section must be present in config")

        # Output directory
        self.output_dir = self.vllm_config.get('output_dir', './vllm_outputs')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Prompt configuration
        self.prompt_template = self.vllm_config.get(
            'prompt_template',
            'Describe this overhead satellite or aerial image. '
            'Identify key features, land use, structures, and any notable patterns.'
        )

        # Sampling parameters
        temperature = self.vllm_config.get('temperature', 0.7)
        max_tokens = self.vllm_config.get('max_tokens', 512)
        top_p = self.vllm_config.get('top_p', 0.9)
        top_k = self.vllm_config.get('top_k', -1)

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens
        )

        self.batch_size = config.get('batch_size', 1)

        # Model initialization parameters
        tensor_parallel_size = config.get('tensor_parallel_size', 1)
        gpu_memory_utilization = config.get('gpu_memory_utilization', 0.9)

        # Initialize vLLM model
        print(f"Loading vLLM model: {self.model_name}")
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True  # Required for some models
        )
        print("Model loaded successfully!")

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

    def _format_prompt(self, image_path: str, custom_prompt: Optional[str] = None) -> str:
        """
        Format the prompt for the model, optionally including image-specific context.

        Args:
            image_path: Path to the image file.
            custom_prompt: Optional custom prompt to override template.

        Returns:
            Formatted prompt string.
        """
        if custom_prompt:
            return custom_prompt

        # Extract filename for context
        filename = os.path.basename(image_path)

        # Use the template
        return self.prompt_template

    def __call__(self, infer_df=None, prompts: Optional[List[str]] = None):
        """
        Run vLLM inference on a set of images.

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

        # Process images in batches
        for batch_start in range(0, len(image_paths), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]

            print(f"Processing batch {batch_start // self.batch_size + 1} "
                  f"({batch_start + 1}-{batch_end}/{len(image_paths)})")

            # Prepare inputs for this batch
            batch_images = []
            batch_prompts = []
            batch_metadata = []

            for idx, img_path in enumerate(batch_paths):
                # Load image and metadata
                img, proj, gt = self._load_image(img_path)
                batch_images.append(img)

                # Get prompt
                if prompts is not None:
                    prompt = prompts[batch_start + idx]
                else:
                    prompt = self._format_prompt(img_path)
                batch_prompts.append(prompt)

                # Store metadata
                batch_metadata.append({
                    'projection': proj,
                    'geotransform': gt
                })

            # Create vLLM inputs
            # For vision models, the format is typically:
            # {"prompt": text, "multi_modal_data": {"image": image}}
            vllm_inputs = []
            for prompt, img in zip(batch_prompts, batch_images):
                vllm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": img}
                })

            # Run inference
            outputs = self.model.generate(vllm_inputs, self.sampling_params)

            # Collect results
            for img_path, prompt, output, metadata in zip(
                batch_paths, batch_prompts, outputs, batch_metadata
            ):
                result = {
                    'image_path': img_path,
                    'prompt': prompt,
                    'response': output.outputs[0].text,
                    'metadata': metadata
                }
                results.append(result)

                # Save individual result as JSON
                output_filename = os.path.splitext(os.path.basename(img_path))[0] + '_vllm.json'
                output_path = os.path.join(self.output_dir, output_filename)
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)

                print(f"Processed: {img_path}")
                print(f"Response: {result['response'][:200]}...")  # Print first 200 chars

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
        config: The loaded configuration dict for vLLM inference.

    Returns:
        DataFrame containing at least one column: 'image' with paths to image files.
    """
    infer_csv = config.get('inference_data_csv')
    if not infer_csv:
        raise ValueError("'inference_data_csv' must be specified in config")

    infer_df = get_data_paths(infer_csv, infer=True)
    return infer_df
