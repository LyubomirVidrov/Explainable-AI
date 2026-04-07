import matplotlib.pyplot as plt


def display_masked_image(dataset_sub, masked_images_grad_cam, masked_images_rise):
    '''
    Display masked images in a subplot grid. 

    Args: 
        dataset_sub (torch.utils.data.dataset.Subset): Images
        masked_images_grad_cam (list[numpy.ndarray]): List of images where the most important 
        pixels have been removed. 
        masked_images_rise (list[numpy.ndarray]): List of images where the most important 
        pixels have been removed. 
    '''
    
    _, ax = plt.subplots(nrows=4, ncols=3, figsize=(20, 15))

    col_titles = ["Original", "GradCAM", "RISE"]

    # column headers
    for j in range(3):
        ax[0][j].set_title(col_titles[j], fontsize=18, pad=20)

    for i in range(len(dataset_sub)):
        image, _ = dataset_sub[i]

        ax[i][0].imshow(image)
        ax[i][0].axis('off')

        ax[i][1].imshow(masked_images_grad_cam[i])
        ax[i][1].axis('off')

        ax[i][2].imshow(masked_images_rise[i])
        ax[i][2].axis('off')
    
    plt.tight_layout()
    plt.show()


def scatter_plot(ious, faithfulness_scores):
    """
    Create a scatter plot showing the relationship between IoU values and faithfulness scores. 
    
    Args:
        ious (list[numpy.float64]): List of intersection over union values (IoU). 
        faithfulness_scores (list[float]): List of faithfulness scores (change in confidence).

    """

    plt.scatter(ious, faithfulness_scores, alpha=0.8, color='blue')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Scatter plot')
    plt.xlabel('IOU')
    plt.ylabel('Faithfulness')
    plt.show()