import matplotlib.pyplot as plt
import numpy as np
import constant as C
def plot_images(images, labels):
    fig, axarr = plt.subplots(3,3)
    for i, ax in enumerate(axarr.flatten()):
        ax.imshow(images[i].reshape(C.img_shape),cmap='binary')
        ax.set_xlabel("Lable:{0}".format(np.argmax(labels[i])))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()