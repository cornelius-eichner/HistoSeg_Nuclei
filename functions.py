from pathlib import Path  
import numpy as np


def change_img_to_label_path(path, fol_pre = 'tissue_images', fol_post = 'mask_binary', suf_pre = 'tif', suf_post = 'png'):
    """
    Helper function to rename image paths to match label paths
    
    INPUT:
    path: Path to image file; pathlib Path object
    fol_pre: folder of image data, to be exchanged with label folder; string
    fol_post: folder of label data, to be inserted in 'path'; string
    suf_pre: file extension of image data; string
    suf_post: file extension of label data; string
    
    RETURN:
    new_path: new path, pointing to corresponding label file; pathlib Path object
    """
    
    parts = list(path.parts)  # get all directories within the path
    parts[parts.index(fol_pre)] = fol_post  # Replace tissue_images with mask_binary
    
    # Get Filename of Path by string splitting the last entry 
    filename = parts[-1].split(".")
    # Replace Filename Suffix
    filename[filename.index(suf_pre)] = suf_post
    
    #Put Together all parts to create the list new_path, pointing to the label directory
    new_path = parts[:-1] + [str(filename[0]) + "." + str(filename[1])]

    return Path(*new_path)  # Combine list back into a Path object

def standardize(img, means, stds):
    """
    Channel wise z transformation to standardize imgages within the sample 

    INPUT:
    img: image data for z-transformation; (WxHxC) numpy array
    means: channel wise means of z-transformation; (3,) numpy array
    stds: channel wise standard deviations of z-transformation; (3,) numpy array
    
    RETURNS: 
    img_normalized: image data with reversed z-transformation; (WxHxC) numpy array

    """
    
    # Create an empty array similar to the image dataset
    img_normalized = np.zeros_like(img)
    
    # Loop over channels, to perform channel-wise z-transformation using external mean and standard deviations 
    for i in range(img.shape[-1]):
        img_normalized[...,i] = (img[...,i] - means[...,i]) / stds[...,i]

    
    return img_normalized

def un_standardize(img, means, stds):
    """
    Helper Function to reverse the channel wise z-transformation of the data preprocessing
    
    INPUT:
    img: standardized image data; (WxHxC) numpy array
    means: channel wise means of pre-applied z-transformation; (3,) numpy array
    stds: channel wise standard deviations of pre-applied z-transformation; (3,) numpy array
    
    RETURNS: 
    img_un_normalized: image data with reversed z-transformation; (WxHxC) numpy array
    
    """
    
    # Create empty array
    img_un_normalized = np.zeros_like(img)
    
    # Go through cannels and reverse z-transform
    for i in range(img.shape[-1]):
        img_un_normalized[...,i] = stds[...,i] * img[...,i] + means[...,i]
        
    # Clip data between 0 and 1 to prevent matplotlib errors
    return np.clip(img_un_normalized, 0, 1)

     
def confusion_matrix(X_label, X_pred, print_tab = False):
    """
    Helper Function to calculate and display a binary confusion matrix
    
    INPUT:
    X_label: Label Data; boolean numpy array (nd) 
    X_pred:  Prediction Data; boolean numpy array (nd) 
    print_tab: print the confusion matrix inline: boolean
    
    RETURN: 
    """
    
    # Reshape input variables
    X_label = np.array(X_label).astype(bool).ravel()
    X_pred = np.array(X_pred).astype(bool).ravel()
    
    # Total Pixels
    n_tot = X_label.shape[0]
    
    # Label == True Pixels
    n_lab = X_label.sum()
    
    # Label == False Pixels
    n_no_lab = n_tot - n_lab
    
    # Pred = True Pixels
    n_pred = X_pred.sum()
    
    # Pred = False Pixels
    n_no_pred = n_tot - n_pred
        
    # True Positives
    tp = (X_label * X_pred).sum()
    
    # True Negatives
    tn = (np.logical_not(X_label) * np.logical_not(X_pred)).sum()
    
    # False Positives
    # True Predictions - True Positives
    fp = n_pred - tp
    
    # False Negatives
    # No Predictions - True negatives
    fn = n_no_pred - tn
    
    if print_tab:
        print('               Pred TRUE \t Pred FALSE')
        print('Label TRUE  \t {:.2f} \t \t {:.2f} \t \t {:.2f}'.format(tp/n_tot,fn/n_tot,n_lab/n_tot))
        print('Label FALSE \t {:.2f} \t \t {:.2f} \t \t {:.2f}'.format(fp/n_tot,tn/n_tot,n_no_lab/n_tot))
        print('                 {:.2f} \t\t {:.2f} '.format(n_pred/n_tot,n_no_pred/n_tot))        
    
    return n_tot, n_lab, n_no_lab, n_pred, n_no_pred, np.array([[tp, fp],[fn, tn]]), np.array([[tp, fp],[fn, tn]])/n_tot


