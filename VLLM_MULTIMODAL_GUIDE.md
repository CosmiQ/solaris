# vLLM Multimodal Support for Solaris

This guide explains how to use vision-language models (VLMs) with Solaris for multimodal analysis of overhead imagery.

## Overview

Solaris now supports **vLLM-based multimodal inference**, enabling you to use state-of-the-art vision-language models to analyze satellite and aerial imagery with natural language. This allows you to:

- Generate detailed descriptions of overhead imagery
- Answer questions about specific features in images
- Classify land use and detect objects using natural language
- Perform complex visual reasoning on geospatial data
- Extract structured information from satellite/aerial photos

## Supported Models

Any vLLM-compatible vision-language model can be used, including:

- **LLaVA** (various versions): `llava-hf/llava-1.5-7b-hf`, `llava-hf/llava-1.5-13b-hf`, `llava-hf/llava-v1.6-mistral-7b-hf`
- **Qwen-VL**: `Qwen/Qwen-VL-Chat`
- **Other multimodal models** supported by vLLM

## Installation

### Option 1: Conda Environment (Recommended)

vLLM requires **Python 3.8+** and **PyTorch 2.0+**, which are newer than Solaris's default dependencies. We provide a separate environment file:

```bash
# Create a new conda environment with vLLM support
conda env create -f environment-vllm.yml

# Activate the environment
conda activate solaris-vllm

# Install Solaris in development mode
pip install -e .
```

### Option 2: pip with vLLM Requirements

```bash
# Create a Python 3.8+ virtual environment
python3.8 -m venv venv-vllm
source venv-vllm/bin/activate

# Install vLLM-compatible dependencies
pip install -r requirements-vllm.txt

# Install Solaris
pip install -e .
```

### GPU Requirements

- **GPU Memory**: At least 16GB VRAM recommended for 7B models, 24GB+ for 13B models
- **CUDA**: 11.8 or later
- **Multiple GPUs**: Supported via tensor parallelism (see configuration below)

## Quick Start

### 1. Prepare Your Data

Create a CSV file listing your images:

```csv
image
/path/to/satellite_image_1.tif
/path/to/satellite_image_2.tif
/path/to/aerial_photo.jpg
```

### 2. Create a Configuration File

Create a YAML configuration file (e.g., `vllm_config.yml`):

```yaml
model_name: 'llava-hf/llava-1.5-7b-hf'
batch_size: 1
tensor_parallel_size: 1
gpu_memory_utilization: 0.9
inference_data_csv: 'images.csv'

vllm_inference:
  output_dir: './vllm_outputs'
  prompt_template: 'Describe this satellite image in detail.'
  temperature: 0.7
  max_tokens: 512
  top_p: 0.9
```

### 3. Run Inference

```bash
# Basic usage
solaris_vllm_infer --config vllm_config.yml

# With custom prompt
solaris_vllm_infer --config vllm_config.yml --prompt "What type of buildings are visible?"

# Single image quick analysis
solaris_vllm_infer --config vllm_config.yml --image /path/to/image.tif

# Verbose output
solaris_vllm_infer --config vllm_config.yml --verbose
```

## Configuration Reference

### Model Configuration

```yaml
# Model identifier (HuggingFace ID or local path)
model_name: 'llava-hf/llava-1.5-7b-hf'

# Batch size for processing multiple images
batch_size: 1

# Number of GPUs for tensor parallelism (1, 2, 4, 8)
tensor_parallel_size: 1

# GPU memory utilization (0.0 to 1.0)
gpu_memory_utilization: 0.9
```

### Inference Configuration

```yaml
vllm_inference:
  # Output directory for JSON results
  output_dir: './vllm_outputs'

  # Prompt template - can include placeholders
  prompt_template: |
    Analyze this overhead image and describe:
    1. Terrain and landscape type
    2. Built structures and infrastructure
    3. Vegetation patterns
    4. Notable features

  # Sampling parameters
  temperature: 0.7    # 0.0-1.0, higher = more creative
  max_tokens: 512     # Maximum response length
  top_p: 0.9          # Nucleus sampling
  top_k: -1           # Top-k sampling (-1 = disabled)
```

### Input Data

```yaml
# CSV file with 'image' column
inference_data_csv: 'images.csv'
```

## Usage Examples

### Example 1: Land Use Classification

```yaml
model_name: 'llava-hf/llava-1.5-7b-hf'
batch_size: 1
inference_data_csv: 'satellite_images.csv'

vllm_inference:
  output_dir: './landuse_classification'
  prompt_template: |
    Classify the land use in this satellite image.
    Categories: Urban, Agricultural, Forest, Water, Barren.
    Provide percentages and describe key features.
  temperature: 0.3  # Lower for more consistent classification
  max_tokens: 256
```

### Example 2: Building Detection

```yaml
model_name: 'llava-hf/llava-1.5-13b-hf'
batch_size: 1
inference_data_csv: 'urban_images.csv'

vllm_inference:
  output_dir: './building_analysis'
  prompt_template: |
    Analyze the buildings in this aerial image:
    - Count approximate number of buildings
    - Describe building types (residential, commercial, industrial)
    - Note building density and layout patterns
    - Identify any notable structures
  temperature: 0.5
  max_tokens: 400
```

### Example 3: Change Detection

```yaml
model_name: 'llava-hf/llava-v1.6-mistral-7b-hf'
batch_size: 1
inference_data_csv: 'temporal_images.csv'

vllm_inference:
  output_dir: './change_detection'
  prompt_template: |
    Examine this satellite image for signs of recent development or change:
    - New construction or infrastructure
    - Vegetation changes
    - Land use transitions
    - Environmental impacts
  temperature: 0.6
  max_tokens: 512
```

## Multi-GPU Inference

For faster processing with multiple GPUs:

```yaml
model_name: 'llava-hf/llava-1.5-13b-hf'
tensor_parallel_size: 4  # Use 4 GPUs
batch_size: 2
gpu_memory_utilization: 0.85

# ... rest of config
```

## Output Format

Results are saved as JSON files in the output directory:

```json
{
  "image_path": "/path/to/image.tif",
  "prompt": "Describe this satellite image in detail.",
  "response": "This satellite image shows an urban area with...",
  "metadata": {
    "projection": "PROJCS[...]",
    "geotransform": [...]
  }
}
```

Additionally, an `all_results.json` file contains all results in a single array.

## Python API

You can also use the vLLM inference programmatically:

```python
from solaris.nets.vllm_infer import VLLMInferer
import pandas as pd

# Create configuration
config = {
    'model_name': 'llava-hf/llava-1.5-7b-hf',
    'batch_size': 1,
    'inference_data_csv': 'images.csv',
    'vllm_inference': {
        'output_dir': './outputs',
        'prompt_template': 'Describe this image.',
        'temperature': 0.7,
        'max_tokens': 512
    }
}

# Initialize inferer
inferer = VLLMInferer(config)

# Run inference
results = inferer()

# Process results
for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Response: {result['response']}")
    print()
```

### Custom Prompts Per Image

```python
# Create custom prompts for each image
df = pd.read_csv('images.csv')
prompts = [
    "What type of terrain is this?",
    "How many buildings are visible?",
    "Describe the vegetation patterns."
]

# Run with custom prompts
results = inferer(infer_df=df, prompts=prompts)
```

## Performance Tips

1. **Batch Size**: Increase for better GPU utilization, but watch memory usage
2. **Tensor Parallelism**: Use multiple GPUs for larger models or faster inference
3. **GPU Memory**: Adjust `gpu_memory_utilization` if you encounter OOM errors
4. **Model Selection**: Smaller models (7B) are faster, larger models (13B) are more accurate
5. **Temperature**: Lower values (0.1-0.3) for factual tasks, higher (0.7-0.9) for creative descriptions

## Troubleshooting

### Out of Memory Errors

```yaml
# Reduce memory usage
gpu_memory_utilization: 0.7  # Lower from 0.9
batch_size: 1                # Reduce batch size
# Or use a smaller model
model_name: 'llava-hf/llava-1.5-7b-hf'  # Instead of 13b
```

### CUDA Version Mismatch

Ensure CUDA 11.8+ is installed:
```bash
nvidia-smi  # Check CUDA version
```

If needed, install compatible PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Model Download Issues

Models are downloaded from HuggingFace on first use. For offline use:

```bash
# Download model ahead of time
from huggingface_hub import snapshot_download
snapshot_download('llava-hf/llava-1.5-7b-hf')
```

## Examples

See the `examples/vllm/` directory for complete example configurations:

- `vllm_config_llava.yml` - Basic LLaVA configuration
- `vllm_config_custom.yml` - Advanced configuration with custom settings
- `images.csv` - Example image list format

## Integration with Existing Solaris Workflows

vLLM inference is designed to complement existing Solaris functionality:

1. **Pre-processing**: Use Solaris tiling and preprocessing before vLLM inference
2. **Post-processing**: Use vLLM descriptions to augment vector/raster outputs
3. **Evaluation**: Combine traditional metrics with LLM-based semantic analysis

Example workflow:
```bash
# 1. Tile large images
make_masks --config tiling_config.yml

# 2. Run traditional CNN model
solaris_run_ml --config segmentation_config.yml

# 3. Run vLLM analysis
solaris_vllm_infer --config vllm_config.yml

# 4. Combine results for comprehensive analysis
```

## Citation

If you use Solaris with vLLM in your research, please cite:

- Solaris: [Original Solaris citation]
- vLLM: [vLLM paper/repository]
- The specific vision-language model you used (e.g., LLaVA)

## Contributing

We welcome contributions! Please see the main Solaris README for contribution guidelines.

## Support

For issues specific to vLLM integration, please file an issue on the Solaris GitHub repository with the `vllm` label.

## License

Apache-2.0 (same as Solaris)
