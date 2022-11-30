# Open-set recognition using Outlier Exposure

Paper implementation of the article Outlier Exposure for Open Set Crop Recognition from Multitemporal Image Sequences. 

This work evaluates the auxiliary method, named as Outlier Exposure, for multitemporal semantic segmentation of farmland crops. In this problem, we use Campo Verde and Luis Eduardo Magalhaes as in-distribution (ID) and out-of-distribution (OOD) data. 

In order to use Outlier Exposure in this kind of problem, we adapt the loss function $L$ as $L = L_{ID} + \lambda_{OE} L_{OE}$, where $L_{ID}$ represents the loss function for known classes (in this work, Focal loss), and $L_{OE}$ is the loss related to the OOD training data. 

## How to use 

1. Prepare data for the problem. Both datasets (Campo Verde and Luis Eduardo Magalhaes) should be placed in `datasets` folder with proper alias (`datasets\lm` for Luis Eduardo Magalhaes and `datasets\cv` for Campo Verde). The user can also try a different pair of ID/OOD dataset, but some changes may be required (e.g. change task configuration in `params_train` folder and adapt all of the preprocessing involved in this work).

2. Run `train_and_evaluate.py` to train the Fully convolutional recurrent network using Outlier Exposure. 

3. Run `evaluate_open_set.py` to evaluate the trained network in test set, reporting visual results and metrics for OSR.

## Requirements 

This project requires some Python libraries. Please use the `requirements.txt` to install all required packages. Also, the datasets for multitemporal semantic segmentation of farmland crops are necessary to run this experiment. Please contact us for data acquisition. 
