USD AAI-521: Applied Computer Vision for AI
Final Project

AI Image Enhancement Web Application

<br>1 - Implementation of Enhancement Tasks:
<br>Image Denoising:
<br>• Load a pre-trained denoising model from Hugging Face.
<br>• Fine-tune the model on the noisy image dataset.
<br>• Evaluate and refine the model performance.
<br>Image Super-Resolution:
<br>• Load a pre-trained super-resolution model from Hugging Face.
<br>• Fine-tune the model on the low-resolution image dataset.
<br>• Evaluate and refine the model performance.
<br>Image Colorization:
<br>• Load a pre-trained colorization model from Hugging Face.
<br>• Fine-tune the model on the grayscale image dataset.
<br>• Evaluate and refine the model performance.
<br>Image Inpainting:
<br>• Load a pre-trained inpainting model from Hugging Face.
<br>• Fine-tune the model on the damaged image dataset.
<br>• Evaluate and refine the model performance.
<br>
<br>2 - Integration and Deployment:
<br>• Integrate the fine-tuned models into a single system.
<br>• Develop a user interface (UI) for users to upload images and select
<br>enhancement tasks.
<br>• Deploy the system as a web application using frameworks like Flask or
<br>Django.
<br>
<br>3 - Evaluation and Testing:
<br>• Evaluate the system using the test dataset.
<br>• Conduct user testing to gather feedback and make improvements.

# Application

# Build the image

docker-compose build

# Run the container

docker-compose up

# Run in detached mode

docker-compose up -d

# View logs

docker-compose logs -f

# Stop

docker-compose down

# Check point

curl <http://localhost:5000/health>

# Colorize image

curl -X POST \
 -F "image=@/path/to/grayscale_image.png" \
 -F "strength=0.75" \
 -F "guidance_scale=7.5" \
 -F "num_inference_steps=50" \
 <http://localhost:5000/api/colorize> \
 --output colorized.png

# Inpaint Image

curl -X POST \
 -F "image=@/path/to/image.png" \
 -F "mask_top=0.25" \
 -F "mask_bottom=0.75" \
 -F "mask_left=0.25" \
 -F "mask_right=0.75" \
 -F "prompt=fill in the missing parts realistically" \
 <http://localhost:5000/api/inpaint> \
 --output inpainted.png

# Convert to Grayscale

curl -X POST \
 -F "image=@/path/to/color_image.png" \
 <http://localhost:5000/api/grayscale> \
 --output grayscale.png
