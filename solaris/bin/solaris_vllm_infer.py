"""
CLI entry point for LiteLLM multimodal inference in Solaris.

This command enables vision-language model inference on overhead imagery
using US-based models (Google Gemini) and NVIDIA models via LiteLLM.
"""

import argparse
import os
import sys
from ..utils.config import parse
from ..nets.litellm_infer import MultimodalInferer


def main():
    """Main entry point for LiteLLM multimodal inference CLI."""
    parser = argparse.ArgumentParser(
        description='Run LiteLLM multimodal inference on overhead imagery using Gemini and NVIDIA models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with Gemini
  solaris_vllm_infer --config litellm_config.yml

  # Inference with custom prompt
  solaris_vllm_infer --config litellm_config.yml --prompt "What buildings are visible in this image?"

  # Quick inference on a single image
  solaris_vllm_infer --config litellm_config.yml --image path/to/image.tif

Supported Models:
  - gemini/gemini-1.5-pro (Google Gemini Pro with vision)
  - gemini/gemini-1.5-flash (Google Gemini Flash - faster)
  - gemini/gemini-pro-vision (Google Gemini Vision)
  - NVIDIA models via nvidia/* prefix

Environment Variables:
  GEMINI_API_KEY - Google Gemini API key (get from https://makersuite.google.com/app/apikey)
  NVIDIA_API_KEY - NVIDIA API key (if using NVIDIA models)

For more information, see the documentation at:
https://github.com/CosmiQ/solaris
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help="Full path to a YAML-formatted config file specifying "
             "parameters for LiteLLM multimodal inference."
    )

    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default=None,
        help="Optional custom prompt to override the config template. "
             "This will be used for all images."
    )

    parser.add_argument(
        '--image', '-i',
        type=str,
        default=None,
        help="Optional path to a single image for quick inference. "
             "If provided, overrides inference_data_csv in config."
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Optional output directory override. "
             "If provided, overrides output_dir in config."
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output."
    )

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Load config
    try:
        config = parse(args.config)
    except Exception as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

    # Handle single image mode
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)

        # Create a temporary CSV for single image inference
        import pandas as pd
        import tempfile

        temp_csv = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        temp_csv_path = temp_csv.name
        temp_csv.close()

        # Write single image to CSV
        df = pd.DataFrame({'image': [args.image]})
        df.to_csv(temp_csv_path, index=False)
        config['inference_data_csv'] = temp_csv_path

        if args.verbose:
            print(f"Processing single image: {args.image}")

    # Override output directory if provided
    if args.output:
        if 'litellm_inference' not in config:
            config['litellm_inference'] = {}
        config['litellm_inference']['output_dir'] = args.output

    # Override prompt if provided
    if args.prompt:
        if 'litellm_inference' not in config:
            config['litellm_inference'] = {}
        config['litellm_inference']['prompt_template'] = args.prompt

    # Run inference
    try:
        if args.verbose:
            print("Initializing LiteLLM multimodal model...")

        inferer = MultimodalInferer(config)

        if args.verbose:
            print("Running inference...")

        results = inferer()

        if args.verbose:
            print(f"\nSuccess! Processed {len(results)} image(s)")
            print(f"Results saved to: {inferer.output_dir}")

        # Clean up temp CSV if in single image mode
        if args.image:
            try:
                os.unlink(temp_csv_path)
            except Exception:
                pass  # Ignore cleanup errors

    except Exception as e:
        print(f"Error during inference: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
