# LiteLLM Multimodal Support for Solaris

This guide explains how to use vision-language models with Solaris for multimodal analysis of overhead imagery using **US-based models** (Google Gemini) and **NVIDIA models** via LiteLLM.

## Overview

Solaris now supports **LiteLLM-based multimodal inference**, enabling you to use state-of-the-art vision-language models from trusted US-based providers to analyze satellite and aerial imagery with natural language. This allows you to:

- Generate detailed descriptions of overhead imagery
- Answer questions about specific features in images
- Classify land use and detect objects using natural language
- Perform complex visual reasoning on geospatial data
- Extract structured information from satellite/aerial photos

## Supported Models

### Google Gemini (US-based, Recommended)

- **gemini/gemini-1.5-flash** - Fast and efficient, great for most tasks
- **gemini/gemini-1.5-pro** - Most capable, best for complex analysis
- **gemini/gemini-pro-vision** - Earlier version with vision capabilities

### NVIDIA Models

- Various vision-language models available through NVIDIA API catalog
- Check [https://build.nvidia.com/](https://build.nvidia.com/) for current offerings

### Why These Models?

- **US-based**: Google Gemini is developed and operated in the United States
- **Security**: Enterprise-grade security and compliance
- **Performance**: State-of-the-art multimodal capabilities
- **Reliability**: Production-ready APIs with high availability
- **Cost-effective**: Competitive pricing, especially Gemini Flash

## Installation

### Option 1: pip Installation (Recommended)

```bash
# Install Solaris with LiteLLM support
pip install -r requirements-litellm.txt
pip install -e .

# Or install just the additional packages
pip install litellm google-generativeai
```

### Option 2: Conda Environment

```bash
# Create a new conda environment with LiteLLM support
conda env create -f environment-litellm.yml

# Activate the environment
conda activate solaris-litellm

# Install Solaris in development mode
pip install -e .
```

### API Key Setup

#### Google Gemini (Recommended)

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the environment variable:

```bash
export GEMINI_API_KEY='your-gemini-api-key-here'
```

Or add to your `.bashrc` / `.zshrc`:

```bash
echo 'export GEMINI_API_KEY="your-gemini-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

#### NVIDIA Models

1. Get your API key from [NVIDIA Build](https://build.nvidia.com/)
2. Set the environment variable:

```bash
export NVIDIA_API_KEY='your-nvidia-api-key-here'
```

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

Create a YAML configuration file (e.g., `litellm_config.yml`):

```yaml
model_name: 'gemini/gemini-1.5-flash'
inference_data_csv: 'images.csv'

litellm_inference:
  output_dir: './gemini_outputs'
  prompt_template: 'Describe this satellite image in detail.'
  temperature: 0.7
  max_tokens: 512
  top_p: 0.9
```

### 3. Run Inference

```bash
# Make sure API key is set
export GEMINI_API_KEY='your-api-key'

# Basic usage
solaris_vllm_infer --config litellm_config.yml

# With custom prompt
solaris_vllm_infer --config litellm_config.yml --prompt "What type of buildings are visible?"

# Single image quick analysis
solaris_vllm_infer --config litellm_config.yml --image /path/to/image.tif

# Verbose output
solaris_vllm_infer --config litellm_config.yml --verbose
```

## Configuration Reference

### Model Configuration

```yaml
# Model identifier (LiteLLM format)
model_name: 'gemini/gemini-1.5-flash'

# Options:
# - 'gemini/gemini-1.5-flash' (Fast, cost-effective)
# - 'gemini/gemini-1.5-pro' (Most capable)
# - 'gemini/gemini-pro-vision' (Earlier version)
# - 'nvidia/model-name' (NVIDIA models)

# API key (optional if set as environment variable)
# api_key: 'your-api-key-here'

# Batch size (usually 1 for API-based inference)
batch_size: 1
```

### Inference Configuration

```yaml
litellm_inference:
  # Output directory for JSON results
  output_dir: './litellm_outputs'

  # Prompt template
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
```

### Input Data

```yaml
# CSV file with 'image' column
inference_data_csv: 'images.csv'

# Enable verbose logging (optional)
verbose: false
```

## Usage Examples

### Example 1: Land Use Classification with Gemini Flash

```yaml
model_name: 'gemini/gemini-1.5-flash'
inference_data_csv: 'satellite_images.csv'

litellm_inference:
  output_dir: './landuse_classification'
  prompt_template: |
    Classify the land use in this satellite image.
    Categories: Urban, Agricultural, Forest, Water, Barren.
    Provide percentages and describe key features.
  temperature: 0.3  # Lower for more consistent classification
  max_tokens: 256
```

### Example 2: Building Detection with Gemini Pro

```yaml
model_name: 'gemini/gemini-1.5-pro'
inference_data_csv: 'urban_images.csv'

litellm_inference:
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

### Example 3: Infrastructure Assessment

```yaml
model_name: 'gemini/gemini-1.5-flash'
inference_data_csv: 'infrastructure_images.csv'

litellm_inference:
  output_dir: './infrastructure_analysis'
  prompt_template: |
    Identify and describe infrastructure in this overhead image:
    - Roads and highways (types, conditions)
    - Bridges and overpasses
    - Power lines and utilities
    - Industrial facilities
    - Transportation hubs
    Provide a structured assessment.
  temperature: 0.4
  max_tokens: 512
```

### Example 4: Environmental Monitoring

```yaml
model_name: 'gemini/gemini-1.5-pro'
inference_data_csv: 'environmental_sites.csv'

litellm_inference:
  output_dir: './environmental_monitoring'
  prompt_template: |
    Analyze this satellite image for environmental assessment:
    - Vegetation health and coverage
    - Water bodies and quality indicators
    - Signs of erosion or degradation
    - Human impact on natural areas
    - Conservation status
  temperature: 0.5
  max_tokens: 600
```

## Output Format

Results are saved as JSON files in the output directory:

```json
{
  "image_path": "/path/to/image.tif",
  "prompt": "Describe this satellite image in detail.",
  "response": "This satellite image shows an urban area with dense residential development...",
  "metadata": {
    "projection": "PROJCS[...]",
    "geotransform": [...]
  }
}
```

Additionally, an `all_results.json` file contains all results in a single array.

## Python API

You can also use the LiteLLM inference programmatically:

```python
from solaris.nets.litellm_infer import MultimodalInferer
import pandas as pd
import os

# Set API key
os.environ['GEMINI_API_KEY'] = 'your-api-key'

# Create configuration
config = {
    'model_name': 'gemini/gemini-1.5-flash',
    'batch_size': 1,
    'inference_data_csv': 'images.csv',
    'litellm_inference': {
        'output_dir': './outputs',
        'prompt_template': 'Describe this overhead image.',
        'temperature': 0.7,
        'max_tokens': 512
    }
}

# Initialize inferer
inferer = MultimodalInferer(config)

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

## Model Comparison

| Model | Speed | Cost | Capability | Best For |
|-------|-------|------|------------|----------|
| gemini-1.5-flash | ⚡⚡⚡ Fast | 💰 Low | ⭐⭐⭐ Good | Most tasks, high volume |
| gemini-1.5-pro | ⚡⚡ Medium | 💰💰 Medium | ⭐⭐⭐⭐⭐ Excellent | Complex analysis, detailed descriptions |
| gemini-pro-vision | ⚡⚡ Medium | 💰💰 Medium | ⭐⭐⭐⭐ Very Good | General multimodal tasks |

**Recommendation**: Start with **gemini-1.5-flash** for most use cases. Upgrade to **gemini-1.5-pro** if you need more detailed analysis or handling of complex scenes.

## Performance Tips

1. **Choose the Right Model**:
   - Gemini Flash for speed and cost-effectiveness
   - Gemini Pro for maximum accuracy and detail

2. **Optimize Prompts**:
   - Be specific about what you want
   - Use structured prompts for consistent outputs
   - Include examples in prompts for better results

3. **Temperature Settings**:
   - 0.1-0.3: Factual, consistent outputs (classification)
   - 0.4-0.6: Balanced (general analysis)
   - 0.7-0.9: Creative descriptions

4. **Batch Processing**:
   - Process multiple images sequentially
   - Consider rate limits for your API tier

5. **Error Handling**:
   - The system automatically handles errors gracefully
   - Check output JSON for error messages

## Troubleshooting

### API Key Issues

```bash
# Verify API key is set
echo $GEMINI_API_KEY

# Test API key with LiteLLM
python -c "import litellm; print(litellm.completion(model='gemini/gemini-1.5-flash', messages=[{'role':'user','content':'Hi'}]))"
```

### Rate Limiting

If you encounter rate limits:
- Reduce batch size
- Add delays between requests
- Upgrade your API tier

### Image Format Issues

Supported formats:
- JPEG, PNG, GIF, WebP
- GeoTIFF (automatically converted)

Large images are automatically resized if needed.

### Model Not Found

```bash
# List available models
python -c "import litellm; print(litellm.model_list)"
```

## Cost Estimation

### Google Gemini Pricing (as of 2024)

- **Gemini 1.5 Flash**: ~$0.075 per 1K images (with text)
- **Gemini 1.5 Pro**: ~$0.25 per 1K images (with text)

*Note: Prices are approximate. Check [Google AI Pricing](https://ai.google.dev/pricing) for current rates.*

### Cost Optimization

- Use Gemini Flash for initial analysis
- Reserve Gemini Pro for images requiring detailed analysis
- Batch similar queries together
- Cache results to avoid re-processing

## Examples

See the `examples/vllm/` directory for complete example configurations:

- `litellm_config_gemini.yml` - Gemini Flash basic configuration
- `litellm_config_gemini_pro.yml` - Gemini Pro advanced configuration
- `litellm_config_nvidia.yml` - NVIDIA models configuration
- `example_litellm_inference.py` - Python API examples
- `images.csv` - Example image list format

## Integration with Existing Solaris Workflows

LiteLLM inference complements existing Solaris functionality:

1. **Pre-processing**: Use Solaris tiling before LiteLLM inference
2. **Post-processing**: Augment vector/raster outputs with LLM descriptions
3. **Evaluation**: Combine traditional metrics with semantic analysis

Example workflow:
```bash
# 1. Tile large images
make_masks --config tiling_config.yml

# 2. Run traditional CNN model
solaris_run_ml --config segmentation_config.yml

# 3. Run LiteLLM analysis for semantic understanding
solaris_vllm_infer --config litellm_config_gemini.yml

# 4. Combine results for comprehensive analysis
```

## Security & Privacy

- API keys are never logged or stored
- Images are sent to provider APIs (Google, NVIDIA)
- For sensitive data, consider:
  - On-premises deployment (if available from provider)
  - Image preprocessing to remove sensitive information
  - Using private API endpoints (enterprise plans)

## Citation

If you use Solaris with LiteLLM in your research, please cite:

- Solaris: [Original Solaris citation]
- LiteLLM: [LiteLLM repository]
- Google Gemini: [Gemini technical report]
- NVIDIA models: [Specific model documentation]

## Contributing

We welcome contributions! Please see the main Solaris README for contribution guidelines.

## Support

For issues specific to LiteLLM integration:
- File an issue on the Solaris GitHub repository
- Tag with `litellm` or `multimodal` labels

For API-specific issues:
- Google Gemini: [Google AI Support](https://ai.google.dev/support)
- NVIDIA: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

## License

Apache-2.0 (same as Solaris)

---

**Ready to get started?** Set up your Gemini API key and try the quick start example above!
