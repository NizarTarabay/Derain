'''
implementation of the "Single-Image-Based Rain and Snow RemovalUsing Multi-guided FilterXianhui Zheng1,
Yinghao Liao,WeiGuo, Xueyang Fu, and Xinghao Ding" paper
'''

import cv2
import numpy as np
import argparse


def sharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=2.0, threshold=0):
    '''
    Return a sharpened version of the image, using an unsharp mask.
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    :param image: ndarray
        Input image to be sharpen
    :param kernel_size: tuple
        The size of the kernel
    :param sigma: float
        Gaussian sigma
    :param amount: float
        Amount of sharpness
    :param threshold: int
    :return: ndarray
        shrpened image
    '''
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    # Sharpen the image from the blurred image
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def guidedFilter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """
    # Image smoothing
    meanGuidanceImage = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    meanInputImage = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    correctionGuidanceImage = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    correctionGuidanceInputImage = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))

    varI = correctionGuidanceImage - meanGuidanceImage ** 2
    covIp = correctionGuidanceInputImage - meanGuidanceImage * meanInputImage  # the covariance of (i, p) in each local patch

    # "the relationship among I, p, and q given by (5), (6), and (8) are indeed in the form of image filtering"
    a = covIp / (varI + e)  # Eqn. (5) in "Guided Image Filtering Kaiming He1, Jian Sun2, and Xiaoou Tang" paper;
    b = meanInputImage - a * meanGuidanceImage  # Eqn. (6) in "Guided Image Filtering Kaiming He1, Jian Sun2, and Xiaoou Tang" paper

    meanA = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    meanB = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = meanA * i + meanB  # Eqn. (8) in "Guided Image Filtering Kaiming He1, Jian Sun2, and Xiaoou Tang" paper
    return q


def derain (input_path, output_path):
    '''
    Implementation of the derain function represented in the paper cited above
    :param input_path: string
        The path to the input image
    :param output_path: string
        The path to the saved output path
    '''
    # Input rain or snow image I_in
    I_in = cv2.imread(input_path)

    # Low frequency part I_lf
    I_lf = cv2.boxFilter(I_in, cv2.CV_64F, (7, 7))
    I_lf = I_lf.astype('float64') / 255

    # High frequency part I_hf
    I_hf = abs(I_in.astype('float64') / 255 - I_lf)
    I_hf = I_hf + 0.5

    # Edge enhancement method1
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    edge_enhance = cv2.filter2D(I_lf * 255, -1, kernel)

    # Edge enhancement method 2 (comment method 1 and uncomment method 2)
    # edge_enhance = sharp_mask(I_lf * 255, kernel_size=(5, 5), sigma=1.0, amount=2, threshold=0)

    edge_enhance = edge_enhance.astype('float64') / 255

    # Guided filter
    guided_filtered = guidedFilter(edge_enhance, I_hf, 20, 0.003)

    # Recovery image I_r
    recovery = (guided_filtered + I_lf)

    # Minimum I_in and I_r -> I_cr
    I_min = np.minimum(recovery, I_in.astype('float64') / 255)

    # Weighted summation equation 9 in the paper, betta = 0.5
    ref = 0.5 * I_min + (1 - 0.5) * recovery

    # Guided filter -> refine recovered image I_rr
    guided_filtered2 = guidedFilter(I_min, ref, 20, 0.003)

    # Edge enhancement
    guided_filtered2_enhance = sharp_mask(guided_filtered2 * 255, kernel_size=(5, 5), sigma=3.0, amount=2.0,
                                          threshold=0)

    # Plot and save the results
    cv2.imshow('low frequency', I_lf)
    cv2.imshow('high frequency', I_hf)
    cv2.imshow('first guided filter', guided_filtered)
    cv2.imshow('mean edge enhance', edge_enhance)
    cv2.imshow('recovery', recovery)
    cv2.imshow('min (between I_in and I_r)', I_min)
    cv2.imshow('result', guided_filtered2_enhance)

    # Save the output image
    cv2.imwrite(output_path, guided_filtered2_enhance)
    cv2.waitKey()


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
args = parser.parse_args()


if __name__ == '__main__':
    if args.input is None:
        derain('image/t047_frame1.png', 'image/t047_frame1_derained.png')
    else:
        derain(args.input, args.output)