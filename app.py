import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import List

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Set the model to training mode
pipeline.unet.train()
pipeline.text_encoder.train()
pipeline.vae.eval()

# Load the COCO dataset
coco_dataset = load_dataset("phiyodr/coco2017", split="train")

# Preprocess the dataset
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def preprocess_data(example):
    if example['captions']:
        # Process multiple captions if available
        captions = example['captions'][:3]  # Take up to 3 captions per image
        all_tokens = []
        for caption in captions:
            tokens = tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=50,
                return_tensors="pt"
            )
            all_tokens.append(tokens.input_ids[0].tolist())
        # Pad if fewer than 3 captions
        while len(all_tokens) < 3:
            all_tokens.append([0] * 50)
        return {"input_ids": all_tokens}
    else:
        return {"input_ids": [[0] * 50] * 3}

coco_dataset = coco_dataset.map(preprocess_data, remove_columns=coco_dataset.column_names)

# Set up DataLoader
def collate_fn(batch):
    all_input_ids = []
    for item in batch:
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in item['input_ids']]
        all_input_ids.extend(input_ids)
    return torch.stack(all_input_ids)

# Reduce batch size and account for multiple captions
train_dataloader = DataLoader(coco_dataset, batch_size=15, shuffle=True, drop_last=True, collate_fn=collate_fn)

# Set up optimizer
optimizer = AdamW(list(pipeline.unet.parameters()) + list(pipeline.text_encoder.parameters()), lr=5e-6)

# Training loop
num_epochs = 1
accumulation_steps = 4

for epoch in range(num_epochs):
    total_loss = 0
    optimizer.zero_grad()

    for step, input_ids in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        input_ids = input_ids.to(device, dtype=torch.long)

        try:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Get text embeddings for all prompts
                text_embeddings = pipeline.text_encoder(input_ids)[0]

                # Sample noise for each embedding
                noise = torch.randn((input_ids.shape[0], 4, 64, 64), dtype=torch.float16, device=device)

                # Get timesteps
                timesteps = torch.ones((input_ids.shape[0],), device=device, dtype=torch.long)

                # Forward pass
                noise_pred = pipeline.unet(
                    sample=noise,
                    timestep=timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample

                # Compute loss
                loss = F.mse_loss(noise_pred, noise) / accumulation_steps

            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item() * accumulation_steps

            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory error. Reducing batch size or model size may help.")
                torch.cuda.empty_cache()
                continue
            print(f"Error encountered: {e}")
            raise

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

# Save the fine-tuned model
pipeline.save_pretrained("./fine-tuned-stable-diffusion-coco")
print("Training completed and model saved.")

def generate_images(
    pipeline: StableDiffusionPipeline,
    prompts: List[str],
    num_images_per_prompt: int = 1,
    output_dir: str = "generated_images"
):
    """
    Generate multiple images for multiple prompts

    Args:
        pipeline: The Stable Diffusion pipeline
        prompts: List of text prompts
        num_images_per_prompt: Number of images to generate per prompt
        output_dir: Directory to save the generated images
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        try:
            print(f"Generating images for prompt: {prompt}")
            with torch.no_grad():
                # Generate multiple images for each prompt
                for j in range(num_images_per_prompt):
                    image = pipeline(
                        prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5
                    ).images[0]

                    # Save the image with a unique name
                    image_path = os.path.join(
                        output_dir,
                        f"prompt_{i+1}image{j+1}.png"
                    )
                    image.save(image_path)
                    print(f"Saved image: {image_path}")

        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {str(e)}")
            continue

# Example usage of the generation function
test_prompts = [
    "A beautiful sunset over the mountains",
    "A futuristic city with flying cars",
    "A peaceful garden with blooming flowers",
    "An underwater scene with coral reefs"
]

# Generate 2 images per prompt
generate_images(
    pipeline=pipeline,
    prompts=test_prompts,
    num_images_per_prompt=2,
    output_dir="multi_prompt_images"
)

print("All images generated successfully!")
