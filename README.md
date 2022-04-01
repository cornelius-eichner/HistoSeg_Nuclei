# Histology-NucleiSeg
This repository lays out a deep learning approach for cellular nuclei instance segmentation of histology images stained with hematoxylin and eosin. For this, a U-Net deep learning architecture is coded in PyTorch and trained on provided histology images [Ronneberger O, 2015]. Two U-Nets, with different architecture (number of layers), are developed, trained, and evaluated. 

The scripts perform the following steps:
1. 01-Data_Inspection.ipynb loads the data and checks for potential size conflicts as well as sampling biases
2. 02-Data_Preprocessing.ipynb normalizes, standardizes, and saves the dataset in NumPy format
3. 03-Dataset.ipynb generates a dataset class, capable of random data augmentation
4. 04-UNet-Model.ipynb defines the architectures of the two U-Nets 
5. 05-Train_Model.ipynb trains the model on the smaller U-Net (with fewer layers), and validates the results using nuclei counting and boolean pixel analysis
6. 06-Train_Model_UNet_Full.ipynb trains the model on the full U-Net (as described in Ronneberger et al), and validates the results using nuclei counting and boolean pixel analysis

## Python Environment
An [Anaconda Miniconda](https://docs.conda.io/en/latest/miniconda.html) python 3.7 environment suitable for executing the code is provided in the file 'pytorchenv.yml'. 
To install the environment including all required libraries please run the following command:
`conda env create -f pytorchenv.yml`

The anaconda pytorchenv environment needs to be activated for the code to run. To activate the environment, please run the following command:
`conda activate pytorchenv`

The provided files will be executed in a Jupyter notebook environment, using python 3.7. 


## Literature
Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28

