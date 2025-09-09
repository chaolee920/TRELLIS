# External API Usage Guide for time-check.py

## Overview
The `time-check.py` file has been modified to use **Meshy AI**, one of the best external text-to-3D generation APIs, instead of running local TRELLIS pipelines.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements_external_api.txt
```

### 2. Get Meshy AI API Key
1. Sign up at [Meshy AI](https://www.meshy.ai/)
2. Navigate to your API settings
3. Generate an API key

### 3. Set Environment Variable
```bash
export MESHY_API_KEY="your_api_key_here"
```

## Usage
```bash
python3 time-check.py
```

## Key Changes Made

### Advantages of External API:
- **No GPU Required**: Runs on any machine without CUDA/GPU requirements
- **Higher Quality**: Meshy AI uses state-of-the-art 3D generation models
- **Reduced Memory**: No need to load large models locally
- **Better Scalability**: Can handle multiple requests without hardware constraints

### API Features Used:
- **Preview Mode**: Fast generation for initial results
- **Refine Mode**: High-quality generation for better results
- **Multiple Art Styles**: Cartoon and realistic styles for variety
- **PLY Output**: Compatible with existing validation pipeline

### How It Works:
1. Creates two concurrent tasks using Meshy AI:
   - Standard text-to-3D with cartoon style (preview mode)
   - Refined text-to-3D with realistic style (refine mode)
2. Downloads PLY files from completed tasks
3. Uses existing validation system to score results
4. Implements early termination logic (same as before)

## Cost Considerations
- Meshy AI charges per generation
- Preview mode is typically cheaper than refine mode
- Consider costs when running multiple concurrent tasks

## Troubleshooting
- Ensure `MESHY_API_KEY` is set correctly
- Check internet connection for API calls
- Monitor API rate limits and quotas
- Refined mode tasks take longer (up to 15 minutes)
