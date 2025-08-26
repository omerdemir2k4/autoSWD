# LOOCV for Attention-ResUNet

## Data Preperation
After downloading the EDF and Excel files, one could extract the normalized 1024-datapoint long amplitude values at 100 Hz as .npy files in **X** headed files, and 0 or 1 values in **Y** headed files in a single directory by running the **`loocv_data_prep.py`**.

## Running LOOCV
Using the output directory of **`loocv_data_prep.py`** directory as the input in the last line of **`loocv_unet.py`**, one could run LOOCV and create the results, including accuracy, F1 score, precision, recall, specificity, balanced accuracy, ROC AUC, average precision. Model is provided in **`model_unet`**.
