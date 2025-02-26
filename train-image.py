# Step 1: Install required libraries
!pip install diffusers transformers accelerate torch torchvision --quiet

# Step 2: Import necessary modules
import os
import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display
from PIL import Image

# Step 3: Define model path
model_path = "/content/stable-diffusion-v1-5"

# Step 4: Check if the model directory exists; if not, download the model
if not os.path.exists(model_path):
    print("Model directory does not exist! Re-downloading...")
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.save_pretrained(model_path)

# Step 5: Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained(model_path).to(device)

# Step 6: Define the prompt for generating the gabion wall image
prompt = (
    "A modern two-story house with a gabion stone wall fence in the foreground. "
    "The fence consists of metal wire cages filled with irregularly shaped gray and brown stones, "
    "with horizontal wooden slats between sections. The house features a contemporary design with a "
    "sloped roof, dark brown and white exterior, large glass windows, and a balcony with a glass railing. "
    "The setting includes a lush green lawn, a stone pathway, and a cloudy sky."
)

# Step 7: Define a negative prompt to avoid unwanted artifacts
negative_prompt = (
    "blurry details, unrealistic textures, low quality, missing stones, plastic-looking wall, collapsed fence, wire mesh missing."
)

# Step 8: Generate the image
print("Generating image... Please wait.")
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=7.5).images[0]

# Step 9: Save the image
image_path = "/content/gabion_wall.png"
image.save(image_path)

# Step 10: Display the generated image
display(image)

# Step 11: Provide download link for the image
print(f"Download the image here: {image_path}")
