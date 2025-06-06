from tensorflow.keras.preprocessing.image import save_img

def save_image(image_tensor, filename):
    # Convert the EagerTensor to a NumPy array
    numpy_array = image_tensor.numpy()

    # Save the NumPy array as a PNG image
    save_img(filename, numpy_array)