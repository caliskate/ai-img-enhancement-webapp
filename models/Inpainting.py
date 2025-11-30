import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

import os

# Path for your fine-tuned inpainting model
FINE_TUNED_INPAINT_DIR = "./models/inpainting_pipeline"

# Global pipeline so it loads once
inpaint_pipe = None


def load_inpainting_pipeline():
    """
    Loads the fine-tuned Stable Diffusion Inpainting pipeline.
    Falls back to base model if LoRA/FT weights are not available.
    """

    global inpaint_pipe

    if inpaint_pipe is not None:
        return inpaint_pipe

    # Check for fine-tuned weights
    if os.path.exists(FINE_TUNED_INPAINT_DIR):
        print(
            f"[Inpainting] Loading fine-tuned model from {FINE_TUNED_INPAINT_DIR}"
        )
        model_to_load = FINE_TUNED_INPAINT_DIR
    else:
        print(
            "[Inpainting] Fine-tuned inpaint model not found. Using base SD-inpaint model."
        )
        model_to_load = "runwayml/stable-diffusion-inpainting"

    # Load pipeline
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_to_load,
        torch_dtype=torch.float16
        if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False)

    # Move to GPU if available
    if torch.cuda.is_available():
        inpaint_pipe = inpaint_pipe.to("cuda")
        print("[Inpainting] Using GPU")
    else:
        print("[Inpainting] Using CPU")

    return inpaint_pipe


def preprocess_mask(mask: Image.Image) -> Image.Image:
    """
    Converts mask to binary (white=fill, black=keep) and resizes.
    """
    mask = mask.convert("L")  # 1-channel
    mask = mask.resize((512, 512))

    # Convert to binary mask (threshold)
    mask = mask.point(lambda p: 255 if p > 50 else 0)

    return mask


def preprocess_inpaint_image(img: Image.Image) -> Image.Image:
    """
    Resizes and ensures 3-channel RGB image for pipeline.
    """
    img = img.convert("RGB")
    img = img.resize((512, 512))
    return img


def inpaint_image(input_img: Image.Image,
                  mask_img: Image.Image) -> Image.Image:
    """
    Main inpainting function.
    Accepts PIL images → returns PIL image.
    The mask defines areas to fill: WHITE = fill, BLACK = keep.
    """

    if input_img is None:
        raise ValueError("No input image provided for inpainting.")

    if mask_img is None:
        raise ValueError("No mask provided for inpainting.")

    pipe = load_inpainting_pipeline()

    image = preprocess_inpaint_image(input_img)
    mask = preprocess_mask(mask_img)

    prompt = "A realistic reconstruction of the missing regions of the photo."

    result = pipe(prompt=prompt,
                  image=image,
                  mask_image=mask,
                  num_inference_steps=40,
                  guidance_scale=7.5)

    output = result.images[0]
    return output
