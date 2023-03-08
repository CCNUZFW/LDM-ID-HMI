# A Unified Interpretable Intelligent Learning Diagnosis Framework for Learning Performance Prediction in Intelligent Tutoring Systems

Source code and data set for the paper *A Unified Interpretable Intelligent Learning Diagnosis Framework for Learning Performance Prediction in Intelligent Tutoring Systems*.

The code is the implementation of LDM model.

## Citations:
@article{Wang2023,<br>
  title = {A Unified Interpretable Intelligent Learning Diagnosis Framework for Learning Performance Prediction in Intelligent Tutoring Systems},<br>
  author = {Wang, Zhifeng and Yan, Wenxing and Zeng, Chunyan and Tian, Yuan and Dong, Shi},<br>
  year = {2023},<br>
  journal = {International Journal of Intelligent Systems},<br>
  volume = {2023},<br>
  pages = {1--20},<br>
}<br>

## Dependencies:

- python 3.6
- pytorch >= 1.0 (pytorch 0.4 might be OK but pytorch<0.4 is not applicable)
- numpy
- json
- sklearn



## Usage


Run the model:

`python train.py {device} {epoch}`

For example:

`python train.py cuda:0 5`  or `python train.py cpu 5`




