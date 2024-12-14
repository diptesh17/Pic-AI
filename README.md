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

## Features
- A beautiful sunset over the mountains : 

![1111](https://github.com/user-attachments/assets/c942c318-d943-4798-94f1-48e6047431b2)

![111111](https://github.com/user-attachments/assets/694a9a5e-40f5-4b27-b6ff-38878f88adbc)

- A futuristic city with flying cars :

![222](https://github.com/user-attachments/assets/dc71ab63-44f2-4e57-b147-ad25de8d06ca)

![22](https://github.com/user-attachments/assets/518649f0-9744-459d-a194-eefd8d69717e)

- A peaceful garden with blooming flowers :
  
![33](https://github.com/user-attachments/assets/304e2967-4044-4c35-a695-d5df9c74f1ae)

![333](https://github.com/user-attachments/assets/7def7ea7-3018-4739-980c-52a74526a2b9)

- An underwater scene with coral reefs :

![4](https://github.com/user-attachments/assets/3f117727-770f-40c8-b14d-8a630567b7af)

![444](https://github.com/user-attachments/assets/4a52b1be-d467-4b9e-8a4e-b8b3c12e8e85)
