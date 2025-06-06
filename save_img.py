
def tensor_to_png(tensor, filename):
    # Ensure the tensor is in the correct format for PNG encoding
    tensor = tf.cast(tensor, tf.uint8)

    # Encode the tensor as a PNG image
    png_image = tf.io.encode_png(tensor)

    # Save the PNG image to a file
    with open(filename, 'wb') as f:
        f.write(png_image.numpy())