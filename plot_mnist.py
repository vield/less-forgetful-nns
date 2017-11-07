import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.examples.tutorials.mnist import input_data


def mnist_imshow(img):
    plt.imshow(img.reshape([28, 28]), cmap="gray")
    plt.axis("off")


if __name__ == "__main__":

    sns.set()

    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

    num_images = len(mnist.train._images)
    img = mnist.train._images[2017]

    np.random.seed(1)
    perm = np.random.permutation(mnist.train._images.shape[1])

    img_permuted = img[perm]

    plt.figure()
    mnist_imshow(img)

    plt.figure()
    mnist_imshow(img_permuted)

    plt.show()