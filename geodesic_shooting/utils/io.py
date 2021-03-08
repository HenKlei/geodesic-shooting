import imageio
from PIL import Image


def load_image(path):
    """Loads a greyscale image from a given path.

    Parameters
    ----------
    path
        Path to look at for the image.

    Returns
    -------
    The converted image.
    """
    image = imageio.imread(path)
    image_grey = image[:, :, 0]
    return image_grey / 255.


def save_image(image, path):
    """Saves an image to the given path.

    Parameters
    ----------
    image
        Image to save.
    path
        Path to save the image at.
    """
    image = Image.fromarray((image * 255).astype('uint8'))
    imageio.imsave(path, image)


def save_animation(images, path):
    """Creates a gif out of the given images.

    Parameters
    ----------
    path
        Path to save the animation at.
    images
        Sequence of images to include in the animation.
    """
    images = [Image.fromarray((images[t] * 255).astype('uint8')) for t in range(len(images))]
    imageio.mimsave(path, images)
