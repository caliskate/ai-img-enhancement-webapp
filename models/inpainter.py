import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


class InpainterModel:
    def __init__(self, model_path=None, device=None):
        """
        Initialize inpainting model
        Args:
            model_path: Path to model
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or "runwayml/stable-diffusion-inpainting"
        self.pipe = None

    def load_model(self):
        """Load the model pipeline"""
        print(f"Loading inpainting model from {self.model_path}...")

        dtype = torch.float16 if self.device == 'cuda' else torch.float32

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            safety_checker=None
        )

        self.pipe = self.pipe.to(self.device)

        # Enable memory optimizations
        if self.device == 'cuda':
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

        print("Inpainting model loaded successfully!")
        return self

    def inpaint(self, image, mask, prompt="fill in the missing parts realistically",
                guidance_scale=7.5, num_inference_steps=50):
        """
        Inpaint masked regions of an image
        Args:
            image: PIL Image (original)
            mask: PIL Image (mask - white regions will be inpainted)
            prompt: Text prompt for inpainting
            guidance_scale: How closely to follow prompt
            num_inference_steps: Number of denoising steps
        Returns:
            PIL Image (inpainted)
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure images are RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if mask.mode != 'RGB':
            mask = mask.convert('RGB')

        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
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
