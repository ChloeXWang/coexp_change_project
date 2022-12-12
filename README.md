# Temporally Resolved Gene Co-expressions with Semantic Modelling
This repo is dedicated to the share the final project code for CSC 2611 :star2:

- The model training and validation code can be found in the main folder:
  - co_exp_lineage_wandb_DiffTime.py for the Co-expression Prediction model training and testing
  - co_exp_lineage_wandb_ExpEst.py for the Expression Estimation model training and testing
  - coexp_change_inference.ipynb for time-course construction, change point detection, Pearson correlation calculation and result visualizations
  
- The datasets can be found in the ./datasets folder:
  - ./dyn-LI-2000-1 for the Linear dataset
  - ./dyn-BF-2000-1 for the Birfurcation dataset
  - ./dyn-TF-2000-1 for the Trifurcation dataset
  - The datasets were produced by [Pratapa et al.](https://www.nature.com/articles/s41592-019-0690-6) and retrieved from https://zenodo.org/record/3701939#.Y5a5huzMJwo.
  
- The trained models can be found in the ./models folder: 
  - There are two models associated with each dataset, with a total of six sub-folders
  - The Co-expression Prediction model sub-folders are named with "DiffTime", and the Expression Estimation model sub-folders named with "ExpEstimator"
  - In coexp_change_inference.ipynb, the model from the latest epoch is used for inference
  
- requirements.txt includes all packages needed for this repo

Please feel free to let me know if any issues. Thank you for reviewing! :raised_hands: :smiley:
  
