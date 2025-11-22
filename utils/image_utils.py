from PIL import Image
import io
import base64
import numpy as np


def validate_image(file):
    """Validate uploaded image file"""
    if not file:
        return False, "No file provided"
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    filename = file.filename.lower()
    
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        return False, f"Invalid file type. Allowed: {allowed_extensions}"
    
    try:
        img = Image.open(file.stream)
        img.verify()
        file.stream.seek(0)  # Reset stream after verify
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def resize_to_divisible_by_8(image):
    """Resize image so dimensions are divisible by 8 (SD requirement)"""
    width, height = image.size
    new_width = width - (width % 8)
    new_height = height - (height % 8)
    return image.resize((new_width, new_height), Image.LANCZOS)


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def create_mask_from_coordinates(image_size, coordinates):
    """
    Create a mask from coordinates
    coordinates: dict with 'top', 'bottom', 'left', 'right' as fractions (0-1)
    """
    img_width, img_height = image_size
    
    mask = Image.new("RGB", (img_width, img_height), color=(0, 0, 0))
    
    top = int(img_height * coordinates.get('top', 0.1))
    bottom = int(img_height * coordinates.get('bottom', 0.9))
    left = int(img_width * coordinates.get('left', 0.1))
    right = int(img_width * coordinates.get('right', 0.9))
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left, top, right, bottom], fill=(255, 255, 255))
    
    return mask


def convert_to_grayscale(image):
    """Convert image to grayscale (3-channel RGB for SD compatibility)"""
    return image.convert("L").convert("RGB")
