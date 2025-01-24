
import numpy as np
import cv2



def find_largest_class1_component(image):
    """
    Find the connected component with the largest number of pixels of class 1
    and retain both class 1 and class 2 pixels in that component.

    :param image: numpy array of the image with values 0, 1, and 2
    :return: binary image with only the largest connected component of class 1 and 2
    """
    # Create a binary image where class 1 and class 2 pixels are set to 1
    binary_class1_class2 = ((image == 1) | (image == 2)).astype(np.uint8)

    # Label connected components
    num_labels, labels_im = cv2.connectedComponents(binary_class1_class2)

    # Find the connected component with the largest number of class 1 pixels
    max_size = 0
    best_label = 0
    for label in range(1, num_labels):
        size = np.sum((labels_im == label) & (image == 1))
        if size > max_size:
            max_size = size
            best_label = label

    # Create binary image with only the largest connected component of class 1 and 2
    largest_component = np.zeros_like(image)
    largest_component[labels_im == best_label] = image[labels_im == best_label]

    return largest_component

def process_images(images):
    """
    Process each image in the array to retain only the largest connected component of class 1 and 2.

    :param images: numpy array of shape (2048, 1000, 1, 10)
    :return: numpy array with only the largest connected component of class 1 and 2 in each image
    """
    processed_images = np.zeros_like(images)
    for i in range(images.shape[-1]):
        print(f'Processing image {i}/{images.shape[-1]}......................')
        image = images[:, :, 0, i]
        largest_component = find_largest_class1_component(image)
        processed_images[:, :, 0, i] = largest_component

    return processed_images
