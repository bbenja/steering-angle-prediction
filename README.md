# steering-angle-prediction
Master's thesis on novel algorithms for autonomous driving based on steering angle prediction


<h4>steering-angle-prediction</h4>

Implementation of an autonomous driving algorithm based on Swin Transformer.
Proposed NN takes images of 32x32x3 and returns a value [0-1] representing vehicle's steering angle.
The Swin-TF implementation is based on https://keras.io/examples/vision/swin_transformers/

Implemented custom loss function to put an emphasis on greater steering angles.
Implemented generators that read batches of images and angles from HDF5 files.

Trained and tested on data collected from CARLA and MetaDrive simulators and in respective virtual environments.
