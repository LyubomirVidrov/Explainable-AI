import torch
import numpy as np
import matplotlib.patches as patches
from PIL import Image
from scipy.stats import spearmanr


def make_prediction(image, model, preprocess, weights, device):
    '''
    This function performs image classification inference. 

    Args:
        image (PIL.Image.Image): Original input image. 
        model (torchvision.models.resnet.ResNet): Pretrained classification model.
        preprocess (torchvision.transforms._presets.ImageClassification): Image preprocessing steps. 
        weights (ResNet50_Weights): Pretrained ResNet50 model weights. 
        device (torch.device): Device on which inference is performed.

    Returns:
        prediction (torch.Tensor): model's prediction
        class_id (int): Index of the target class.
        score (float): prediction confidence score
        category_name (str): label
    '''

   # prepare the data for inference 
    batch = preprocess(image).unsqueeze(0).to(device)
    # make prediction
    with torch.no_grad():
        prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()  # class index 
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]

    return prediction, class_id, score, category_name


def return_prediction(prediction, score, category_name, weights):
    '''
    Print the model's prediction along with its top 5 options to choose from. 

    Args:
        prediction (torch.Tensor): model's prediction
        score (float): prediction confidence score
        category_name (str): label 
        weights (ResNet50_Weights): pretrained ResNet-50 (ImageNet weights)
    '''
    
    # Print prediction class and accuracy 
    print(f"{category_name}: {100 * score:.2f}%")

    top5_prob, top5_catid = torch.topk(prediction, 5)
    print('Top 5 -> [', end='')
    # Print the top 5 options
    for j in range(5):
        class_id_t5 = top5_catid[j].item()
        score_t5 = top5_prob[j].item()
        if j != 4:
            print(f'{weights.meta["categories"][class_id_t5]}: {100 * score_t5:.2f}%', end=', ')
        else: 
            print(f'{weights.meta["categories"][class_id_t5]}: {100 * score_t5:.2f}%]')


def get_area_of_interest(target):
    '''
    This function returns the area of interest in the shape of a rectangle. 

    Args:
        target (dict): metadata

    Returns:
        area_of_interest (matplotlib.patches.Rectangle): A bounding box surrounding 
        the object of interest. 
    '''

    # get boundaries
    coordinates = target['annotation']['object'][0]['bndbox']

    width = int(coordinates['xmax']) - int(coordinates['xmin'])
    height = int(coordinates['ymax']) - int(coordinates['ymin'])
    # initialize the object
    area_of_interest = patches.Rectangle((int(coordinates['xmin']), int(coordinates['ymin'])), width, height, 
                           linewidth=2, edgecolor='blue', facecolor='none')

    return area_of_interest


def get_top_20(saliency):
    '''
    This function extarcts the most important regions from a saliency map. 

    Args:
        saliency (numpy.ndarray): Saliency map 

    Returns:
        top_20 (numpy.ndarray): A boolean array indicating the 20% most important pixels. 
    '''

    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # keep the top 20% most important pixels/values of the Grad-CAM heatmap
    threshold = np.percentile(saliency_norm, 80)
    top_20 = saliency_norm >= threshold

    return top_20


def get_area_of_importance(top_20):
    '''
    This function returns the area of importance in the shape of a rectangle, influenced 
    by the XAI model in use. 

    Args:
        top_20 (numpy.ndarray): A boolean array indicating the 20% most important pixels. 

    Returns:
        area_of_importance (matplotlib.patches.Rectangle): A bounding box enclosing 
        the top 20% of pixels.  
    '''
    
    # Get bounding box from mask
    ys, xs = np.where(top_20)
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    # Draw bounding box
    area_of_importance = patches.Rectangle((x_min, y_min),
                (x_max - x_min),
                (y_max - y_min),
                linewidth=2,
                edgecolor='red', facecolor='none')
    
    return area_of_importance


def get_IoU(bbox_A, bbox_B):
    '''
    This function computes the Intersection over Union between the ground truth 
    bounding box and the predicted bounding box, which represents the deemed area of importance.

    Args:
        bbox_A (matplotlib.transforms.Bbox): A bounding box for the area of interest.
        bbox_B (matplotlib.transforms.Bbox): A bounding box for the area of importance.

    Returns:
        iou (numpy.float64): Intersection over Union value.
    '''

    bbox_A = bbox_A.get_points().flatten()
    bbox_B = bbox_B.get_points().flatten()

    xA = max(bbox_A[0], bbox_B[0])
    yA = max(bbox_A[1], bbox_B[1])
    xB = min(bbox_A[2], bbox_B[2])
    yB = min(bbox_A[3], bbox_B[3])
    
    # compute the area of intersection rectangle
    inter_area = (xB - xA) * (yB - yA)
    
    # compute the area of both the prediction and ground-truth rectangles
    bbox_A_area = (bbox_A[2] - bbox_A[0]) * (bbox_A[3] - bbox_A[1])
    bbox_B_area = (bbox_B[2] - bbox_B[0]) * (bbox_B[3] - bbox_B[1])
    
    # Formula for computing intersection over union
    iou = inter_area / float(bbox_A_area + bbox_B_area - inter_area)
    
    # no intersection
    iou = np.maximum(iou, 0)
    
    return iou


def _get_class_confidence(model, input_tensor, class_idx):
    '''
    This function returns the model's confidence in its prediction. 

    Args: 
        model (torchvision.models.resnet.ResNet): Pretrained classification model.
        input_tensor (torch.Tensor): Preprocessed image.
        class_idx (int): Index of the target class.

    Returns:
        float: The model's confidence in the specified class. 
    '''
    
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
    
    return probs[0, class_idx].item()


def compute_faithfulness(image, image_np, mask, class_idx, model, preprocess, device):
    '''
    This function computes the faithfulness of an explanation by measuring the drop
    in model confidence after masking the most important regions identified by the XAI model in use. 

    Args:
        image (PIL.Image.Image): Original input image. 
        image_np (numpy.ndarray): Image as a NumPy array.  
        mask (numpy.ndarray): A boolean array indicating the 20% most important pixels. 
        class_idx (int): Index of the target class.
        model (torchvision.models.resnet.ResNet): Pretrained classification model.
        preprocess (torchvision.transforms._presets.ImageClassification): Image preprocessing steps. 
        device (torch.device): Device on which inference is performed. 
    
    Returns:
        faithfulness (float): Confidence drop after masking. 
        old_confidence (float): Original class confidence. 
        new_confidence (float): Confidence after masking.
        masked (numpy.ndarray): Masked image. 
    '''
    
    old_confidence = _get_class_confidence(model, 
                                          preprocess(image).unsqueeze(0).to(device), 
                                          class_idx
                                         )

    masked = image_np.copy()
    masked[mask] = np.float64(0.5) # 0.0
    
    new_confidence = _get_class_confidence(model, 
                                           preprocess(Image.fromarray((masked * 255).astype(np.uint8))).unsqueeze(0).to(device), 
                                           class_idx
                                          )

    # faithfulness = old_confidence - new_confidence 
    faithfulness = (old_confidence - new_confidence) / old_confidence
    
    return faithfulness, old_confidence, new_confidence, masked


def get_spearman_correlation(ious, faithfulness_scores):
    '''
    This function computes the Spearman correlation coefficient. 

    Args:
        ious (list[numpy.float64]): A list containing Intersection over Union scores. 
        faithfulness_scores (list[float]): A list containing faithfulness scores. 

    Returns:
        corr (numpy.float64): Spearman correlation coefficient.
        p_value (numpy.float64): Statistical significance of the correlation.  
    '''
    
    corr, p_value = spearmanr(ious, faithfulness_scores)

    print(f"Spearman correlation: {corr:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    return corr, p_value