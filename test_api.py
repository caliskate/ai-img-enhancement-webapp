import requests
from PIL import Image
import io

BASE_URL = "http://localhost:5000"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_colorize(image_path):
    """Test colorization"""
    print("Testing colorization...")

    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {
            'strength': 0.75,
            'guidance_scale': 7.5,
            'num_inference_steps': 30
        }

        response = requests.post(
            f"{BASE_URL}/api/colorize",
            files=files,
            data=data
        )

    if response.status_code == 200:
        # Save result
        img = Image.open(io.BytesIO(response.content))
        img.save('test_colorized.png')
        print("✅ Colorization successful! Saved as test_colorized.png")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())
    print()


def test_grayscale(image_path):
    """Test grayscale conversion"""
    print("Testing grayscale conversion...")

    with open(image_path, 'rb') as f:
        files = {'image': f}

        response = requests.post(
            f"{BASE_URL}/api/grayscale",
            files=files
        )

    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        img.save('test_grayscale.png')
        print("✅ Grayscale conversion successful! Saved as test_grayscale.png")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())
    print()


def test_inpaint(image_path):
    """Test inpainting"""
    print("Testing inpainting...")

    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {
            'mask_top': 0.3,
            'mask_bottom': 0.7,
            'mask_left': 0.3,
            'mask_right': 0.7,
            'prompt': 'fill in the missing parts realistically',
            'num_inference_steps': 30
        }

        response = requests.post(
            f"{BASE_URL}/api/inpaint",
            files=files,
            data=data
        )

    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        img.save('test_inpainted.png')
        print("✅ Inpainting successful! Saved as test_inpainted.png")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    test_health()
    test_grayscale(image_path)
    test_colorize(image_path)
    test_inpaint(image_path)

    print("🎉 All tests completed!")
