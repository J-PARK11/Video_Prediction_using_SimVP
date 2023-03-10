import os
import io
import imageio
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox, VBox

# 20장의 Frame을 모두 시각화.
def multi_frame(dataset, path='', dataname=['mmnist','taxibj','kth']):
    # Channel definition by data.
    if dataname == 'mmnist': cmap_param = 'gray'

    # Construct a figure on which we will visualize the images.
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))

    # Plot each of the sequential images for one data example.
    for idx, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(dataset[idx]), cmap=cmap_param)
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")
    
    # Save Figure
    if path != '':
        plt.savefig(path)

# True값과 Pred값 비교.
def comparison(true, pred, path='', dataname=['mmnist','taxibj','kth']):
    # Channel definition by data.
    if dataname == 'mmnist': cmap_param = 'gray'

    # Construct a figure for the original and new frames.
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    # Plot the original frames.
    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(true[idx+10]), cmap=cmap_param)
        ax.set_title(f"True {idx + 11}")
        ax.axis("off")

    # Plot the new frames.
    for idx, ax in enumerate(axes[1]):
        # np.squeeze(pred[idx])*255
        ax.imshow(np.squeeze(pred[idx+10]), cmap=cmap_param)
        ax.set_title(f"Pred {idx + 11}")
        ax.axis("off")

    # Display the figure.
    if path != '':
        plt.savefig(path)


# Create Single Videos
def create_single_video(dataset, path=''):
    videos = []
    frames = dataset
    current = np.squeeze(frames)
    current = current[..., np.newaxis] * np.ones(3)
    current = (current * 255).astype(np.uint8)

    if path != '':
        imageio.mimsave(path, current, format="gif")
    return videos

# Create Multi Videos
def create_multi_video(dataset, n, path=''):
    videos = []
    for i, frames in enumerate(dataset[:n]):
        iter_path = (path[:-4] + str(i+1) + '.gif')
        current = create_single_video(frames, path=iter_path)
        videos.append(current)
    return videos