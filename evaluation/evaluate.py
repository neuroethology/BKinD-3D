import time
import warnings
# import tensorflow
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import sys
sys.path.append('.')
from config.config_reader import parse_args, create_parser

from load_dataloader import load_dataloader

from loss.compute_loss import *
# import visualize

# from utils import Logger, mkdir_p, save_images, save_3d_images, save_multi_images,save_images_2
from utils.model_utils import *
import cv2
import numpy as np
import copy

import get_gt_kpts
import get_model_kpts

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import sklearn



def main():

    args = parse_args(create_parser())

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


    
    
    
def main_worker(gpu, ngpus_per_node, args):
    torch.cuda.set_device(args.gpu)
    train_gt_kpts = get_gt_kpts.main_worker(gpu, ngpus_per_node, args, "train")
    train_model_kpts = get_model_kpts.main_worker(gpu, ngpus_per_node, args, "train")
    train_gt_kpts_flat = train_gt_kpts.reshape(len(train_gt_kpts ), -1)
    train_model_kpts_flat = train_model_kpts.reshape(len(train_model_kpts ), -1)
    
    test_gt_kpts = get_gt_kpts.main_worker(gpu, ngpus_per_node, args, "test")
    test_model_kpts = get_model_kpts.main_worker(gpu, ngpus_per_node, args, "test")
    test_model_kpts_flat = test_model_kpts.reshape(len(test_model_kpts ), -1)
    

    lin = sklearn.linear_model.Ridge() #LinearRegression() #
    lin.fit(train_model_kpts_flat.cpu(), train_gt_kpts_flat.cpu())

    pred_train = lin.predict(train_model_kpts_flat.cpu()).reshape(-1, 17, 3)
    
    pred_test = lin.predict(test_model_kpts_flat.cpu()).reshape(-1, 17, 3)
    
    mean_err_train = mpjpe(torch.from_numpy(pred_train).cuda(0), train_gt_kpts)
    mean_err_train_pmpjpe = p_mpjpe(pred_train, train_gt_kpts.cpu().numpy())
    mean_err_train_nmpjpe = n_mpjpe(pred_train, train_gt_kpts.cpu().numpy())
    
    

    mean_err_test = mpjpe(torch.from_numpy(pred_test).cuda(0), test_gt_kpts)
    mean_err_test_pmpjpe = p_mpjpe(pred_test, test_gt_kpts.cpu().numpy())
    mean_err_test_nmpjpe = n_mpjpe(pred_test, test_gt_kpts.cpu().numpy())
    mean_err_test_mpjpe_shift = mpjpe_translate(pred_test, test_gt_kpts.cpu().numpy())
    
   
    
    print("Train MPJPE:" + str(mean_err_train))
    print("Train error PMPJPE:" + str(mean_err_train_pmpjpe))
    print("Train error NMPJPE:" + str(mean_err_train_nmpjpe))
    print("Test MPJPE:" + str(mean_err_test))
    print("Test MPJPE Shifted:" + str(mean_err_test_mpjpe_shift))
    print("Test error PMPJPE:" + str(mean_err_test_pmpjpe))
    print("Test error NMPJPE:" + str(mean_err_test_nmpjpe))
    
    import IPython; IPython.embed()
    
    
    
################################################################################


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim = 2))

def mpjpe_translate(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    
    t = muX - muY # Translation
    
    predicted_shift = predicted + t

    
    return np.mean(np.linalg.norm(predicted_shift - target, axis = 2))




def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY
    
    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))
    
    
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    err_no_mirror = np.linalg.norm(predicted_aligned - target, axis=2)
    
    predicted_aligned_mirror = predicted_aligned*[[1,1,-1]]
    
    err_mirror = np.linalg.norm(predicted_aligned_mirror - target, axis=2)
    err_mirror = np.where(err_mirror<err_no_mirror, err_mirror, err_no_mirror)
    
    
    # Return MPJPE
    return (np.mean(err_no_mirror), np.mean(err_mirror))


#nmpjpe = shift, scale, reflection, no rotation
#mpjpe = shift, no scale, no reflection, no rotation
def n_mpjpe(predicted, target):
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY
    
    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))
    
    
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    
    t = muX - a*muY # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*predicted + t
    
    err_no_mirror = np.linalg.norm(predicted_aligned - target, axis=2)
    
    predicted_aligned_mirror = predicted_aligned*[[1,1,-1]]
    
    err_mirror = np.linalg.norm(predicted_aligned_mirror - target, axis=2)
    err_mirror = np.where(err_mirror<err_no_mirror, err_mirror, err_no_mirror)
    # Return MPJPE
    
    return (np.mean(err_no_mirror), np.mean(err_mirror))





#########################################################################################
main()
