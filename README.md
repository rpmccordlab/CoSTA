# CoSTA

Manuscript

CoSTA: Unsupervised Convolutional Neural Network Learning for Spatial Transcriptomics Analysis

CoSTA is an analysis approach for understanding spatial gene relationship of spatial transcriptomics. 

## Usage

To reproduce results in Manuscript, please follow instruction in notebook tutorial under the foder of "CoSTA".

  |
  
  
  |----Analyzing_Slide-seq_via_CoSTA.ipynb
  
  |
  
  |----Analyzing_MERFISH_via_CoSTA.ipynb

## Other analyses

Analysis about comparison with SPARK and SpatialDE is in the folder of "analysis"

## Requirements
  ###
  * NumPy
  * SciPy
  * Pandas
  * Scikit-learn
  * NaiveDE
  * Pytorch

## Data description
For processed data provided in the data folder, images are flattened by row. The last two columns "x" and "y" specify the the dimension of the image rather than coordinates.
