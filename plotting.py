import matplotlib.pyplot as plt

def plot_segmentation_results(image, mask, threshold, nbins=256, title="", save=False, save_path=" "):
    """
    Plots the image, its histogram with Otsu's threshold, and the selected foreground mask.

    Parameters:
    - image: The image to be displayed.
    - mask: The binary mask obtained from Otsu's thresholding.
    - threshold: The threshold value determined by Otsu's method.
    - nbins: The number of bins to use for the histogram, should match the nbins parameter in otsu_thresholding.
    - title: The title for the entire plot.
    - save_path: The path where the plot will be saved.
    """
    if save and not save_path:
        raise ValueError("Provide save_path when save=True.")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].hist(image.ravel(), bins=nbins, color="gray")
    axes[1].axvline(threshold, color="red", label=f"Otsu: {threshold:.3f}")
    axes[1].set_xlabel("Intensity")
    axes[1].set_ylabel("Pixel count")
    axes[1].legend()

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Segmentation Mask")
    axes[2].axis("off")

    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=500)
    plt.show()
    return fig, axes