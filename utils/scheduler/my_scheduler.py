import math
import torch


def my_decay_v1(current_epoch,current_lr=0.0001):
    if current_epoch == 10 or current_epoch == 30 or current_epoch == 60:
        lrate = current_lr * 0.1
    else:
        lrate = current_lr
    return lrate