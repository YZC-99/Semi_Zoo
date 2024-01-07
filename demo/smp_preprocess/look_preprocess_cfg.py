from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
import numpy as np
import segmentation_models_pytorch as smp


encoder = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'

preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)
preprocessing_params = smp.encoders.get_preprocessing_params(encoder, ENCODER_WEIGHTS)
# print(preprocessing_fn)
print(preprocessing_params)