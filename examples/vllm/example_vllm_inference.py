#!/usr/bin/env python
"""
Example script demonstrating vLLM multimodal inference with Solaris.

This script shows how to use vision-language models to analyze overhead imagery
using both config files and programmatic approaches.
"""

import os
import pandas as pd
from solaris.nets.vllm_infer import VLLMInferer


def example_basic_inference():
    """Basic example using a config dictionary."""
    print("=" * 80)
    print("Example 1: Basic Inference with Config")
    print("=" * 80)

    # Create configuration
    config = {
        'model_name': 'llava-hf/llava-1.5-7b-hf',
        'batch_size': 1,
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': 0.9,
        'inference_data_csv': 'images.csv',
        'vllm_inference': {
            'output_dir': './vllm_outputs_basic',
            'prompt_template': (
                'Describe this overhead satellite or aerial image. '
                'Identify key features, land use, and notable patterns.'
            ),
            'temperature': 0.7,
            'max_tokens': 512,
            'top_p': 0.9
        }
    }

    # Initialize inferer
    inferer = VLLMInferer(config)

    # Run inference
    results = inferer()

    # Print results
    print(f"\nProcessed {len(results)} images")
    for result in results:
        print(f"\nImage: {result['image_path']}")
        print(f"Response: {result['response'][:200]}...")


def example_custom_prompts():
    """Example with custom prompts for each image."""
    print("\n" + "=" * 80)
    print("Example 2: Custom Prompts Per Image")
    print("=" * 80)

    # Create image DataFrame
    images_df = pd.DataFrame({
        'image': [
            '/path/to/urban_scene.tif',
            '/path/to/agricultural_land.tif',
            '/path/to/forest_area.tif'
        ]
    })

    # Create custom prompts for each image
    custom_prompts = [
        "How many buildings are visible in this urban area? Describe their arrangement.",
        "What crops are being grown? Identify field patterns and irrigation systems.",
        "Classify the forest type and estimate canopy density."
    ]

    # Configuration
    config = {
        'model_name': 'llava-hf/llava-1.5-7b-hf',
        'batch_size': 1,
        'inference_data_csv': 'temp.csv',  # Will be overridden
        'vllm_inference': {
            'output_dir': './vllm_outputs_custom',
            'temperature': 0.6,
            'max_tokens': 300
        }
    }

    # Initialize inferer
    inferer = VLLMInferer(config)

    # Run with custom prompts
    results = inferer(infer_df=images_df, prompts=custom_prompts)

    # Print results
    for result in results:
        print(f"\nImage: {os.path.basename(result['image_path'])}")
        print(f"Question: {result['prompt']}")
        print(f"Answer: {result['response']}")


def example_land_use_classification():
    """Example focused on land use classification."""
    print("\n" + "=" * 80)
    print("Example 3: Land Use Classification")
    print("=" * 80)

    config = {
        'model_name': 'llava-hf/llava-1.5-7b-hf',
        'batch_size': 1,
        'inference_data_csv': 'satellite_images.csv',
        'vllm_inference': {
            'output_dir': './vllm_outputs_landuse',
            'prompt_template': (
                'Classify the land use in this satellite image into one or more '
                'of these categories: Urban/Built-up, Agricultural/Cropland, '
                'Forest/Vegetation, Water bodies, Bare land/Desert, Industrial, '
                'Residential. Provide percentages for each category if possible.'
            ),
            'temperature': 0.3,  # Lower for more consistent classification
            'max_tokens': 256
        }
    }

    inferer = VLLMInferer(config)
    results = inferer()

    # Process results for classification
    print(f"\nLand Use Classification Results:")
    for result in results:
        print(f"\n{os.path.basename(result['image_path'])}:")
        print(result['response'])


def example_batch_processing():
    """Example with batch processing for efficiency."""
    print("\n" + "=" * 80)
    print("Example 4: Batch Processing")
    print("=" * 80)

    config = {
        'model_name': 'llava-hf/llava-1.5-7b-hf',
        'batch_size': 4,  # Process 4 images at once
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': 0.85,
        'inference_data_csv': 'large_image_set.csv',
        'vllm_inference': {
            'output_dir': './vllm_outputs_batch',
            'prompt_template': 'Provide a concise summary of this overhead image.',
            'temperature': 0.7,
            'max_tokens': 200
        }
    }

    inferer = VLLMInferer(config)
    results = inferer()

    print(f"\nBatch processed {len(results)} images efficiently")


def example_multi_gpu():
    """Example using multiple GPUs with tensor parallelism."""
    print("\n" + "=" * 80)
    print("Example 5: Multi-GPU Inference")
    print("=" * 80)

    config = {
        'model_name': 'llava-hf/llava-1.5-13b-hf',  # Larger model
        'batch_size': 2,
        'tensor_parallel_size': 4,  # Use 4 GPUs
        'gpu_memory_utilization': 0.85,
        'inference_data_csv': 'images.csv',
        'vllm_inference': {
            'output_dir': './vllm_outputs_multigpu',
            'prompt_template': (
                'Provide a detailed analysis of this satellite image, including '
                'terrain type, infrastructure, vegetation, and any notable features.'
            ),
            'temperature': 0.7,
            'max_tokens': 512
        }
    }

    inferer = VLLMInferer(config)
    results = inferer()

    print(f"\nProcessed {len(results)} images using 4 GPUs")


def main():
    """Run all examples (comment out as needed)."""
    print("vLLM Multimodal Inference Examples for Solaris")
    print("=" * 80)

    # Uncomment the examples you want to run
    # Note: Make sure you have the required image files and CSV files

    # example_basic_inference()
    # example_custom_prompts()
    # example_land_use_classification()
    # example_batch_processing()
    # example_multi_gpu()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("\nTo run these examples:")
    print("1. Ensure vLLM is installed: pip install vllm")
    print("2. Create your images.csv file with image paths")
    print("3. Uncomment the example(s) you want to run")
    print("4. Adjust paths and configurations as needed")
    print("=" * 80)


if __name__ == '__main__':
    main()
