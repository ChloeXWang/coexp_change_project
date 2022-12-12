# Temporally Resolved Gene Co-expressions with Semantic Modelling
This repo is dedicated to the share the final project code for CSC 2611 :star2:

- The model training and validation code can be found in the main folder:
  - co_exp_lineage_wandb_DiffTime.py for the Co-expression Prediction model training
  - co_exp_lineage_wandb_ExpEst.py for the Expression Estimation model training
  - coexp_change_inference.ipynb for time-course construction, change point detection, Pearson correlation calculation and result visualizations
  
- The datasets can be found in the ./datasets folder:
  - ./dyn-LI-2000-1 for the Linear dataset
  - ./dyn-BF-2000-1 for the Birfurcation dataset
  - ./dyn-TF-2000-1 for the Trifurcation dataset
  - The datasets were produced by Pratapa et al. and retrieved from https://zenodo.org/record/3701939#.Y5a5huzMJwo.
  
- The trained models can be found in the ./models folder: 
  - There are two models associated with each dataset
  - The Co-expression Prediction model is tagged with DiffTime, the Expression Estimation model is tagged with ExpEstimator
  - In coexp_change_inference.ipynb, the model from the latest epoch is used for inference
  
- requirements.txt includes all packages needed for this repo

Please feel free to let me know if any issues. Thank you for reviewing! :raised_hands: :smiley:
  
