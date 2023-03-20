# Video_Prediction_using_SimVP
[HYU CVlab] Research on how to predict future video sequences through deep learning methodology by extracting underlying patters from natural video.

Research on how to predict future video sequences through deep learning methodology by extracting underlying patters from natural video.

Research on a more practical and reasonable Video Prediction Modeling and Application based on "SimVP(CVPR2022)" thesis as the 1st project at HYU CVlab.

## Contents
### Caltech
<p align="left"><img src = "figure/230317_OG_SimVP_caltech_300/multi_f_true0.jpg" width = '600'/></p>
<p align="left"><img src = "figure/230317_OG_SimVP_caltech_300/multi_f_pred0.jpg" width = '600'/></p>
<p align="left"><img src = "figure/230317_OG_SimVP_caltech_300/multi_true_video1.gif" width = '600'/></p>
<p align="left"><img src = "figure/230317_OG_SimVP_caltech_300/multi_pred_video1.gif" width = '600'/></p>

### kth
<p align="left"><img src = "figure/230314_OG_SimVP_kth_1000/multi_f_true.jpg" width = '600'/></p>
<p align="left"><img src = "figure/230314_OG_SimVP_kth_1000/multi_f_pred.jpg" width = '600'/></p>
<p align="left"><img src = "figure/230314_OG_SimVP_kth_1000/multi_true_video1.gif" width = '600'/></p>
<p align="left"><img src = "figure/230314_OG_SimVP_kth_1000/multi_pred_video1.gif" width = '600'/></p>

## Dependencies
* torch
* scikit-image
* numpy
* argparse
* tqdm
* cuda

## Overview

* `API/` contains dataloaders & metrics & recoders
* `main.py` is the executable python file with possible arguments.
* `model.py` contains the model.
* `modules.py` contains the layer & unit cells to construct model.
* `exp.py` is the core file for training, validating, and testing pipelines.
* `utils.py` contains various method & custom function for overall codes.
* `evalutation.py` is the evaluation codes by test dataset   
* `visualization.py` is a comprehensive file that collects methods related to visualization.

---------
