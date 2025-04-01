# Neural Style Transfer with TensorFlow

This project implements a neural style transfer algorithm using TensorFlow and VGG19 to blend the content of one image with the style of another.

## Overview
Neural Style Transfer (NST) combines the content of a photograph with the artistic style of a painting or another image. This implementation uses a pre-trained VGG19 model to extract content and style features, optimizing an output image to minimize the difference between these features.

## Features
- Loads and preprocesses content and style images.
- Uses VGG19 to extract style and content features.
- Applies a custom loss function combining style, content, and total variation losses.
- Saves intermediate and final stylized images.

## Prerequisites
- Python 3.x
- TensorFlow (`pip install tensorflow`)
- NumPy (`pip install numpy`)
- Pillow (`pip install pillow`)

## Usage
1. **Prepare Images**  
   Place your content image (e.g., `pika.jpg`) and style image (e.g., `weather.jpg`) in the project directory.

2. **Run the Script**  
   Update the `content_path` and `style_path` variables in the `main()` function with your image paths:
   ```python
   content_path = 'your_content_image.jpg'
   style_path = 'your_style_image.jpg'