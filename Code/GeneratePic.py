import os
import copy
import numpy as np
import matplotlib.pyplot as plt

def colormap(label, map_type, dataname):
    """
    Generate a colormap for a given label and dataset.

    Parameters:
        label (np.ndarray): The label matrix.
        dataname (str): The name of the dataset.
        map_type (str): The type of colormap ('bin' or 'mul').

    Returns:
        np.ndarray: The colored label matrix.
    """
    if map_type == 'bin':
        if dataname in ['Santa', 'Bay']:
            Color_bar = np.array([[0, 0, 0], [255, 255, 255], [128, 128, 128]])
        else:
            Color_bar = np.array([[0, 0, 0], [255, 255, 255]])
    elif map_type == 'mul':
        Color_bar = np.array([[0, 0, 0], [255, 255, 255], [128, 128, 128], [255, 255, 0], [0, 0, 255]])
    else:
        raise ValueError("Invalid map_type. Choose 'bin' or 'mul'.")

    H, W = label.shape
    Y_color = np.zeros((H, W, 3))

    for i in range(H):
        for j in range(W):
            item = label[i, j]
            if item < len(Color_bar):
                Y_color[i, j, :] = Color_bar[item, :] / 255.0

    return Y_color

def result_pic(label, save_path):
    """
    Save the label matrix as an image.

    Parameters:
        label (np.ndarray): The label matrix with colors.
        save_path (str): The path to save the image.

    Returns:
        None
    """
    dpi = 10
    fig = plt.figure(frameon=False)
    fig.set_size_inches(label.shape[1] * 2.0 / dpi, label.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(label)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

def generate_png(GT, Predict, args):
    """
    Generate and save ground truth and prediction images.

    Parameters:
        GT (np.ndarray): Ground truth labels, [H, w].
        Y_pred (np.ndarray): Predicted labels, [H, w].
        args (Namespace): Argument object with necessary attributes.

    Returns:
        None
    """

    # Initialize the label arrays
    Y_mul_label = copy.deepcopy(Predict)
    Y_bin_label = copy.deepcopy(Predict)

    # Determine label assignment logic
    for x in range(GT.shape[0]):
        for y in range(GT.shape[1]): 
            if GT[x, y] == 1 and Predict[x, y] == 0:
                Y_mul_label[x, y] = 3 # FN (False Negatives)
            elif GT[x, y] == 0 and Predict[x, y] == 1:
                Y_mul_label[x, y] = 4 # FP (False Positives)


    # Apply colormaps
    Y_gt_color  = colormap(label = GT,         map_type = 'bin', dataname = args.target)
    Y_bin_color = colormap(label =Y_bin_label, map_type = 'bin', dataname = args.target)
    Y_mul_color = colormap(label =Y_mul_label, map_type = 'mul', dataname = args.target)

    # Create directory if it doesn't exist
    path_name = os.path.join(args.path, 'result')
    os.makedirs(path_name, exist_ok=True)

    # Save images
    print("    Saving Ground Truth image...")
    result_pic(Y_gt_color, os.path.join(path_name, f'PNG_{args.source}2{args.target}_GT.png'))

    print("    Saving Binary Prediction image...")
    result_pic(Y_bin_color, os.path.join(path_name, f'PNG_{args.source}2{args.target}_Bin_Predict.png'))

    print("    Saving Multi-class Prediction image...")
    result_pic(Y_mul_color, os.path.join(path_name, f'PNG_{args.source}2{args.target}_Mul_Predict.png'))



def ReshapeLabel(Orilabel, Prelabel, Position):

    Y_label = copy.deepcopy(Orilabel)

    # Determine label assignment logic
    for i in range(Position.shape[0]):
        x, y = Position[i, :]
        Y_label[x, y] = Prelabel[i]

    return Y_label

