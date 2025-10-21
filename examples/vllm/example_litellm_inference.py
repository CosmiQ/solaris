#!/usr/bin/env python
"""
Example script demonstrating LiteLLM multimodal inference with Solaris.

This script shows how to use vision-language models (Google Gemini, NVIDIA models)
to analyze overhead imagery using both config files and programmatic approaches.
"""

import os
import pandas as pd
from solaris.nets.litellm_infer import MultimodalInferer


def example_gemini_basic():
    """Basic example using Google Gemini Flash."""
    print("=" * 80)
    print("Example 1: Basic Inference with Google Gemini 1.5 Flash")
    print("=" * 80)

    # Set API key (or use environment variable GEMINI_API_KEY)
    # Get your key from: https://makersuite.google.com/app/apikey

    # Create configuration
    config = {
        'model_name': 'gemini/gemini-1.5-flash',  # Fast, efficient Gemini model
        # 'api_key': 'your-google-api-key',  # Or set GEMINI_API_KEY env var
        'batch_size': 1,
        'inference_data_csv': 'images.csv',
        'litellm_inference': {
            'output_dir': './gemini_outputs_basic',
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
    inferer = MultimodalInferer(config)

    # Run inference
    results = inferer()

    # Print results
    print(f"\nProcessed {len(results)} images")
    for result in results:
        print(f"\nImage: {result['image_path']}")
        print(f"Response: {result['response'][:200]}...")


def example_gemini_pro_custom_prompts():
    """Example with Gemini Pro and custom prompts for each image."""
    print("\n" + "=" * 80)
    print("Example 2: Google Gemini Pro with Custom Prompts")
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
        "How many buildings are visible in this urban area? Describe their arrangement and density.",
        "What type of agriculture is visible? Identify field patterns, crop types if possible, and irrigation systems.",
        "Classify the forest type, estimate canopy density, and identify any clearings or roads."
    ]

    # Configuration with Gemini Pro (more capable)
    config = {
        'model_name': 'gemini/gemini-1.5-pro',
        'batch_size': 1,
        'inference_data_csv': 'temp.csv',  # Will be overridden
        'litellm_inference': {
            'output_dir': './gemini_pro_custom',
            'temperature': 0.6,
            'max_tokens': 400
        }
    }

    # Initialize inferer
    inferer = MultimodalInferer(config)

    # Run with custom prompts
    results = inferer(infer_df=images_df, prompts=custom_prompts)

    # Print results
    for result in results:
        print(f"\nImage: {os.path.basename(result['image_path'])}")
        print(f"Question: {result['prompt']}")
        print(f"Answer: {result['response']}")


def example_land_use_classification():
    """Example focused on land use classification with Gemini."""
    print("\n" + "=" * 80)
    print("Example 3: Land Use Classification with Gemini")
    print("=" * 80)

    config = {
        'model_name': 'gemini/gemini-1.5-flash',
        'batch_size': 1,
        'inference_data_csv': 'satellite_images.csv',
        'litellm_inference': {
            'output_dir': './gemini_landuse',
            'prompt_template': (
                'Classify the land use in this satellite image into one or more '
                'of these categories:\n'
                '- Urban/Built-up\n'
                '- Agricultural/Cropland\n'
                '- Forest/Vegetation\n'
                '- Water bodies\n'
                '- Bare land/Desert\n'
                '- Industrial\n'
                '- Residential\n\n'
                'Provide percentages for each category if possible and describe '
                'the dominant land use.'
            ),
            'temperature': 0.3,  # Lower for more consistent classification
            'max_tokens': 256
        }
    }

    inferer = MultimodalInferer(config)
    results = inferer()

    # Process results for classification
    print(f"\nLand Use Classification Results:")
    for result in results:
        print(f"\n{os.path.basename(result['image_path'])}:")
        print(result['response'])


def example_building_detection():
    """Example focused on building and infrastructure detection."""
    print("\n" + "=" * 80)
    print("Example 4: Building Detection with Gemini Pro")
    print("=" * 80)

    config = {
        'model_name': 'gemini/gemini-1.5-pro',
        'batch_size': 1,
        'inference_data_csv': 'urban_images.csv',
        'litellm_inference': {
            'output_dir': './gemini_buildings',
            'prompt_template': (
                'Analyze the buildings and infrastructure in this aerial image:\n'
                '1. Estimate the number of buildings visible\n'
                '2. Classify building types (residential, commercial, industrial)\n'
                '3. Describe building density and layout patterns\n'
                '4. Identify roads, parking lots, and other infrastructure\n'
                '5. Note any notable or unique structures\n\n'
                'Provide a structured analysis.'
            ),
            'temperature': 0.5,
            'max_tokens': 600
        }
    }

    inferer = MultimodalInferer(config)
    results = inferer()

    print(f"\nBuilding Analysis Results:")
    for result in results:
        print(f"\n{os.path.basename(result['image_path'])}:")
        print(result['response'])


def example_nvidia_model():
    """Example using NVIDIA models (when available)."""
    print("\n" + "=" * 80)
    print("Example 5: NVIDIA Model Inference")
    print("=" * 80)

    # Note: Check NVIDIA's API catalog for actual available vision models
    # This is a template - replace with actual model name

    config = {
        'model_name': 'nvidia/vision-model-name',  # Replace with actual model
        # 'api_key': 'your-nvidia-api-key',  # Or set NVIDIA_API_KEY env var
        'batch_size': 1,
        'inference_data_csv': 'images.csv',
        'litellm_inference': {
            'output_dir': './nvidia_outputs',
            'prompt_template': (
                'Provide a detailed analysis of this satellite image, including '
                'terrain type, infrastructure, vegetation, and any notable features.'
            ),
            'temperature': 0.7,
            'max_tokens': 512
        }
    }

    # Note: Uncomment when you have a valid NVIDIA vision model
    # inferer = MultimodalInferer(config)
    # results = inferer()
    # print(f"\nProcessed {len(results)} images using NVIDIA model")

    print("Note: Update model_name with actual NVIDIA vision model from https://build.nvidia.com/")


def example_env_variables():
    """Example showing how to use environment variables for API keys."""
    print("\n" + "=" * 80)
    print("Example 6: Using Environment Variables for API Keys")
    print("=" * 80)

    # Set API key as environment variable (recommended for security)
    # export GEMINI_API_KEY='your-api-key-here'
    # or
    # os.environ['GEMINI_API_KEY'] = 'your-api-key-here'

    config = {
        'model_name': 'gemini/gemini-1.5-flash',
        # No api_key field needed - will use GEMINI_API_KEY from environment
        'batch_size': 1,
        'inference_data_csv': 'images.csv',
        'litellm_inference': {
            'output_dir': './gemini_outputs_env',
            'prompt_template': 'Describe this overhead image concisely.',
            'temperature': 0.7,
            'max_tokens': 300
        }
    }

    print("Configuration expects GEMINI_API_KEY environment variable to be set.")
    print("Set it with: export GEMINI_API_KEY='your-api-key'")

    # Uncomment to run:
    # inferer = MultimodalInferer(config)
    # results = inferer()


def main():
    """Run all examples (comment out as needed)."""
    print("LiteLLM Multimodal Inference Examples for Solaris")
    print("Using US-based models: Google Gemini and NVIDIA")
    print("=" * 80)

    # Uncomment the examples you want to run
    # Note: Make sure you have:
    # 1. Set GEMINI_API_KEY environment variable (get from https://makersuite.google.com/app/apikey)
    # 2. Created your images.csv file with image paths
    # 3. Installed litellm: pip install litellm google-generativeai

    # example_gemini_basic()
    # example_gemini_pro_custom_prompts()
    # example_land_use_classification()
    # example_building_detection()
    # example_nvidia_model()
    # example_env_variables()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("\nTo run these examples:")
    print("1. Install LiteLLM: pip install litellm google-generativeai")
    print("2. Get Gemini API key: https://makersuite.google.com/app/apikey")
    print("3. Set environment variable: export GEMINI_API_KEY='your-key'")
    print("4. Create your images.csv file with image paths")
    print("5. Uncomment the example(s) you want to run")
    print("6. Adjust paths and configurations as needed")
    print("\nSupported Models:")
    print("- gemini/gemini-1.5-flash (Fast, efficient)")
    print("- gemini/gemini-1.5-pro (Most capable)")
    print("- gemini/gemini-pro-vision (Earlier version)")
    print("- NVIDIA models (check https://build.nvidia.com/)")
    print("=" * 80)


if __name__ == '__main__':
    main()
