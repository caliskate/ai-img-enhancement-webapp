import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
import os


class ColorizerModel:
    def __init__(self, model_path=None, device=None):
        """
        Initialize colorizer model
        Args:
            model_path: Path to fine-tuned model (optional)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or "runwayml/stable-diffusion-v1-5"
        self.pipe = None

    def load_model(self):
        """Load the model pipeline"""
        print(f"Loading colorizer model from {self.model_path}...")

        dtype = torch.float16 if self.device == 'cuda' else torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            safety_checker=None
        )

        self.pipe = self.pipe.to(self.device)

        # Enable memory optimizations
        if self.device == 'cuda':
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

        print("Colorizer model loaded successfully!")
        return self

    def colorize(self, grayscale_image, prompt="colorize the grayscale image",
                 strength=0.75, guidance_scale=7.5, num_inference_steps=50):
        """
        Colorize a grayscale image
        Args:
            grayscale_image: PIL Image (grayscale)
            prompt: Text prompt for colorization
            strength: How much to transform (0-1)
            guidance_scale: How closely to follow prompt
            num_inference_steps: Number of denoising steps
        Returns:
            PIL Image (colorized)
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure image is RGB format (3 channels)
        if grayscale_image.mode != 'RGB':
            grayscale_image = grayscale_image.convert('RGB')

        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                image=grayscale_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]

        return output

    def unload_model(self):
        """Unload model to free memory"""
        if self.pipe:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
