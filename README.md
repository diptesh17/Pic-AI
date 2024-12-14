# Pic-AI : An AI-Powered Image Generation with Stable Diffusion

## Features
- Fine-tune the `Stable Diffusion` model on the **COCO 2017 dataset**.
- Preprocess captions for better text-to-image mapping.
- Generate multiple images per prompt with customizable settings.
- Efficient training with `torch.cuda.amp` for mixed precision.

## How to Execute
1. Install dependencies:
   ```bash
   pip install torch diffusers transformers datasets tqdm
2. Fine-tune the model:
   ```bash
   python app.py
3. Generate images with text prompts:
   ```bash
   python generate.py
