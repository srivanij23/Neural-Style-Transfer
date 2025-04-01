import tensorflow as tf
import numpy as np
import PIL.Image
import time
import functools


def tensor_to_image(tensor):
    """Converts a tensor to an image"""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img, max_dim=512):
    """Loads and preprocesses images"""
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def vgg_layers(layer_names):
    """Creates a VGG model that returns a list of intermediate output values"""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    """Calculates the gram matrix of an input tensor"""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Extracts style and content features"""
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    """Calculates the total loss for style transfer"""
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                          for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_outputs)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                            for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_outputs)

    total_loss = style_loss + content_loss
    return total_loss


@tf.function()
def train_step(image, extractor, style_targets, content_targets, style_weight, content_weight, optimizer):
    """Training step for style transfer"""
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(
            outputs, style_targets, content_targets, style_weight, content_weight)
        # Add total variation loss for smoothness
        loss += tf.image.total_variation(image) * 30

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))
    return loss


def run_style_transfer(content_path, style_path, epochs=10, steps_per_epoch=100):
    """Main function to run style transfer"""
    # Load images
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Define content and style layers
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # Create the model
    extractor = StyleContentModel(style_layers, content_layers)

    # Extract style and content targets
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # Create the optimization image (starting with the content image)
    image = tf.Variable(content_image)

    # Set up optimizer
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # Style transfer parameters
    style_weight = 1e-2
    content_weight = 1e4

    # For storing results
    result_images = []

    # Style transfer optimization loop
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            loss = train_step(image, extractor, style_targets, content_targets,
                              style_weight, content_weight, opt)

        # Convert loss to a scalar value for printing
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy():.4f}")

        # Save current result
        result_images.append(tensor_to_image(image))

    return result_images

# Example usage


def main():
    # Replace with your own content and style image paths
    content_path = 'pika.jpg'  # A photograph
    style_path = 'weather.jpg'      # An artwork

    print("Starting neural style transfer...")

    # Run style transfer
    result_images = run_style_transfer(
        content_path=content_path,
        style_path=style_path,
        epochs=1,
        steps_per_epoch=10
    )

    # Save the final result
    final_image = result_images[-1]
    final_image.save('stylized_image.jpg')

    # Save intermediate results to see the progression
    for i, img in enumerate(result_images):
        img.save(f'stylized_progress_{i+1}.jpg')

    print("Style transfer complete! Results saved as 'stylized_image.jpg'")


if __name__ == "__main__":
    main()
