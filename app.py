from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename
import traceback

from models.colorizer import ColorizerModel
from models.inpainter import InpainterModel
from utils.image_utils import (
    validate_image,
    resize_to_divisible_by_8,
    image_to_base64,
    create_mask_from_coordinates,
    convert_to_grayscale
)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODELS_CACHE = 'models_cache'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODELS_CACHE, exist_ok=True)

# Initialize models (lazy loading)
colorizer = None
inpainter = None


def get_colorizer():
    """Lazy load colorizer model"""
    global colorizer
    if colorizer is None:
        colorizer = ColorizerModel()
        colorizer.load_model()
    return colorizer


def get_inpainter():
    """Lazy load inpainter model"""
    global inpainter
    if inpainter is None:
        inpainter = InpainterModel()
        inpainter.load_model()
    return inpainter


# HTML template for web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Image Enhancement API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #635bff;
            padding-bottom: 10px;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .section h2 {
            color: #635bff;
            margin-top: 0;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        button {
            background-color: #635bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #534bda;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 20px;
        }
        .result img {
            max-width: 100%;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-top: 10px;
        }
        .loading {
            display: none;
            color: #635bff;
            font-weight: bold;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
        .controls {
            margin: 15px 0;
        }
        .controls label {
            display: block;
            margin: 10px 0 5px 0;
            font-weight: bold;
        }
        .controls input[type="range"] {
            width: 100%;
        }
        .controls input[type="number"] {
            width: 100px;
            padding: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Image Enhancement API</h1>
        
        <!-- Health Check -->
        <div class="section">
            <h2>API Status</h2>
            <button onclick="checkHealth()">Check Health</button>
            <div id="health-result" class="result"></div>
        </div>

        <!-- Colorization -->
        <div class="section">
            <h2>Image Colorization</h2>
            <p>Upload a grayscale image to colorize it</p>
            <input type="file" id="colorize-file" accept="image/*">
            
            <div class="controls">
                <label>Strength (0.1-1.0): <span id="strength-value">0.75</span></label>
                <input type="range" id="strength" min="0.1" max="1.0" step="0.05" value="0.75" 
                       oninput="document.getElementById('strength-value').textContent = this.value">
                
                <label>Guidance Scale (1-20): <span id="guidance-value">7.5</span></label>
                <input type="range" id="guidance" min="1" max="20" step="0.5" value="7.5"
                       oninput="document.getElementById('guidance-value').textContent = this.value">
                
                <label>Inference Steps:</label>
                <input type="number" id="steps" min="10" max="100" value="50">
            </div>
            
            <button onclick="colorizeImage()" id="colorize-btn">Colorize</button>
            <div class="loading" id="colorize-loading">Processing...</div>
            <div id="colorize-result" class="result"></div>
        </div>

        <!-- Inpainting -->
        <div class="section">
            <h2>Image Inpainting</h2>
            <p>Upload an image and specify mask coordinates (0-1 range)</p>
            <input type="file" id="inpaint-file" accept="image/*">
            
            <div class="controls">
                <label>Mask Top (0-1): <input type="number" id="mask-top" step="0.01" value="0.25" min="0" max="1"></label>
                <label>Mask Bottom (0-1): <input type="number" id="mask-bottom" step="0.01" value="0.75" min="0" max="1"></label>
                <label>Mask Left (0-1): <input type="number" id="mask-left" step="0.01" value="0.25" min="0" max="1"></label>
                <label>Mask Right (0-1): <input type="number" id="mask-right" step="0.01" value="0.75" min="0" max="1"></label>
                
                <label>Prompt:</label>
                <input type="text" id="inpaint-prompt" value="fill in the missing parts realistically" style="width: 100%; padding: 5px;">
            </div>
            
            <button onclick="inpaintImage()" id="inpaint-btn">Inpaint</button>
            <div class="loading" id="inpaint-loading">Processing...</div>
            <div id="inpaint-result" class="result"></div>
        </div>

        <!-- Convert to Grayscale -->
        <div class="section">
            <h2>Convert to Grayscale</h2>
            <input type="file" id="grayscale-file" accept="image/*">
            <button onclick="convertToGrayscale()">Convert</button>
            <div class="loading" id="grayscale-loading">Processing...</div>
            <div id="grayscale-result" class="result"></div>
        </div>
    </div>

    <script>
        async function checkHealth() {
            const resultDiv = document.getElementById('health-result');
            try {
                const response = await fetch('/health');
                const data = await response.json();
                resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        }

        async function colorizeImage() {
            const fileInput = document.getElementById('colorize-file');
            const button = document.getElementById('colorize-btn');
            const loading = document.getElementById('colorize-loading');
            const resultDiv = document.getElementById('colorize-result');
            
            if (!fileInput.files[0]) {
                alert('Please select a file');
                return;
            }

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            formData.append('strength', document.getElementById('strength').value);
            formData.append('guidance_scale', document.getElementById('guidance').value);
            formData.append('num_inference_steps', document.getElementById('steps').value);

            button.disabled = true;
            loading.style.display = 'block';
            resultDiv.innerHTML = '';

            try {
                const response = await fetch('/api/colorize', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Processing failed');
                }

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                resultDiv.innerHTML = `
                    <h3>Result:</h3>
                    <img src="${url}" alt="Colorized image">
                    <br><a href="${url}" download="colorized.png"><button>Download</button></a>
                `;
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            } finally {
                button.disabled = false;
                loading.style.display = 'none';
            }
        }

        async function inpaintImage() {
            const fileInput = document.getElementById('inpaint-file');
            const button = document.getElementById('inpaint-btn');
            const loading = document.getElementById('inpaint-loading');
            const resultDiv = document.getElementById('inpaint-result');
            
            if (!fileInput.files[0]) {
                alert('Please select a file');
                return;
            }

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            formData.append('mask_top', document.getElementById('mask-top').value);
            formData.append('mask_bottom', document.getElementById('mask-bottom').value);
            formData.append('mask_left', document.getElementById('mask-left').value);
            formData.append('mask_right', document.getElementById('mask-right').value);
            formData.append('prompt', document.getElementById('inpaint-prompt').value);

            button.disabled = true;
            loading.style.display = 'block';
            resultDiv.innerHTML = '';

            try {
                const response = await fetch('/api/inpaint', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Processing failed');
                }

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                resultDiv.innerHTML = `
                    <h3>Result:</h3>
                    <img src="${url}" alt="Inpainted image">
                    <br><a href="${url}" download="inpainted.png"><button>Download</button></a>
                `;
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            } finally {
                button.disabled = false;
                loading.style.display = 'none';
            }
        }

        async function convertToGrayscale() {
            const fileInput = document.getElementById('grayscale-file');
            const loading = document.getElementById('grayscale-loading');
            const resultDiv = document.getElementById('grayscale-result');
            
            if (!fileInput.files[0]) {
                alert('Please select a file');
                return;
            }

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            loading.style.display = 'block';
            resultDiv.innerHTML = '';

            try {
                const response = await fetch('/api/grayscale', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Processing failed');
                }

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                resultDiv.innerHTML = `
                    <h3>Result:</h3>
                    <img src="${url}" alt="Grayscale image">
                    <br><a href="${url}" download="grayscale.png"><button>Download</button></a>
                `;
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            } finally {
                loading.style.display = 'none';
            }
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve web interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    import torch
    return jsonify({
        'status': 'healthy',
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'colorizer_loaded': colorizer is not None,
        'inpainter_loaded': inpainter is not None
    })


@app.route('/api/colorize', methods=['POST'])
def api_colorize():
    """
    Colorize a grayscale image
    Form data:
        - image: image file
        - strength: float (0-1, default 0.75)
        - guidance_scale: float (default 7.5)
        - num_inference_steps: int (default 50)
    """
    try:
        # Validate file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        is_valid, error_msg = validate_image(file)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Load and process image
        image = Image.open(file.stream).convert('RGB')
        image = resize_to_divisible_by_8(image)

        # Convert to grayscale if not already
        grayscale_image = convert_to_grayscale(image)

        # Get parameters
        strength = float(request.form.get('strength', 0.75))
        guidance_scale = float(request.form.get('guidance_scale', 7.5))
        num_inference_steps = int(request.form.get('num_inference_steps', 50))

        # Colorize
        model = get_colorizer()
        colorized = model.colorize(
            grayscale_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )

        # Return image
        img_io = io.BytesIO()
        colorized.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/inpaint', methods=['POST'])
def api_inpaint():
    """
    Inpaint masked regions of an image
    Form data:
        - image: image file
        - mask_top: float (0-1, default 0.25)
        - mask_bottom: float (0-1, default 0.75)
        - mask_left: float (0-1, default 0.25)
        - mask_right: float (0-1, default 0.75)
        - prompt: string (default "fill in the missing parts realistically")
        - guidance_scale: float (default 7.5)
        - num_inference_steps: int (default 50)
    """
    try:
        # Validate file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        is_valid, error_msg = validate_image(file)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Load and process image
        image = Image.open(file.stream).convert('RGB')
        image = resize_to_divisible_by_8(image)

        # Create mask from coordinates
        mask_coords = {
            'top': float(request.form.get('mask_top', 0.25)),
            'bottom': float(request.form.get('mask_bottom', 0.75)),
            'left': float(request.form.get('mask_left', 0.25)),
            'right': float(request.form.get('mask_right', 0.75))
        }
        mask = create_mask_from_coordinates(image.size, mask_coords)

        # Get parameters
        prompt = request.form.get(
            'prompt', 'fill in the missing parts realistically')
        guidance_scale = float(request.form.get('guidance_scale', 7.5))
        num_inference_steps = int(request.form.get('num_inference_steps', 50))

        # Inpaint
        model = get_inpainter()
        inpainted = model.inpaint(
            image,
            mask,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )

        # Return image
        img_io = io.BytesIO()
        inpainted.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/grayscale', methods=['POST'])
def api_grayscale():
    """
    Convert image to grayscale
    Form data:
        - image: image file
    """
    try:
        # Validate file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        is_valid, error_msg = validate_image(file)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Load and convert
        image = Image.open(file.stream).convert('RGB')
        grayscale = convert_to_grayscale(image)

        # Return image
        img_io = io.BytesIO()
        grayscale.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
