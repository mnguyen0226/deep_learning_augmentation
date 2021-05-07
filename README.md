# Image Augmentation for Improvement of Performance of Deep Learning Image Classifiers Project - ECE 4580

## About This Project
- **Problem Statement:** Deep convolutional neural networks have been shown to perform remarkably well in image classification tasks in recent years. However, these networks require extremely large training sets in order to train effectively and avoid overfitting. This problem of generalization can be critical to certain tasks, where it is either not possible or not feasible to obtain a suitably large dataset.
- **Intended Approach:** Image augmentation is the technique of deriving augmented images from an existing training set and creating a new training set from the original and augmented images. Using the technique of image augmentation, deep learning image classifiers can achieve superior performance with smaller datasets, thus alleviating the data collection and labeling problem. We intend to utilize a variety of image augmentation techniques learned in ECE 4580 - Digital Image Processing in Spring 2021.
## About Neural Network Agent:
- ResNet9 train, validate, & save script can be found at: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/classifier/model_train.py.
- ResNet9 model-summary can be found at: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/classifier/model_summary.txt.

## About Image Augmentation Methods
- **Affine:**
    - Flip Vertically & Horizontally: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/affine/flip.py.
    - Rotate: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/affine/rotate.py.
    - Shear Vertically & Horizontally: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/affine/shear.py.
    - Translate: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/affine/translate.py.

- **Canny Edge Detection:** https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/edge_detection/canny_edge.py.

- **Frequency Filters - Gaussian, Ideal, Butterworth:**
    -   Frequency Filter: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/frequency/frequency_filter.py.
        - Bandpass Filter: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/frequency/bandpass.py.
        - Highpass Filter: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/frequency/highpass.py.
        - Lowpass Filter: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/frequency/lowpass.py.

- **Intensity Manipulations:** 
    - Adaptive Median Filter: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/intensity/amf.py.
    - Histogram Equalization: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/intensity/hist_equalization.py.
    - Invert: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/augmentation/intensity/invert.py.
## How To Test Image Augmentation Methods (Recommend Using This For Testing Augmentation Methods For Your Uses)
- Augmentation-method testing script can be found at: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/test_image_aug.py

## How To Augment Image & Train Agents
- Note on set up machine for augmenting and training agents reprodution:
    - CUDA 10.0
    - Pytorch 1.2.0: https://pytorch.org/get-started/previous-versions/.
    - GPU that I use - Cloudera:
![alt text](https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/train_results/Cloudera.PNG)

- Images Augmentation Task:
    - Step 1: Set up the right image path and image-save path in process_image.py.
    - Step 2: Choose the augmentation methods that you prefer in process_image.py
    - Step 3:
        ```
        python preprocess_image.py
        ```
- Training: 
## Results & Comparisons
- Results for training and validating ResNet9 on each image augmentation datasets can be found at: https://github.com/mnguyen0226/image-augmentation-dnn-performance/tree/main/train_results including:
    - "Accuracy vs Each Epoch" graphs.
    - "Learning Rate vs Batch Number" graphs.
    - "Loss Rate vs Each Epoch" graphs.
    - 1 testing image classification in testing batch.
    - Training results after 30 epochs.
    - Training shuffled 64 images batch.
    - Testing 64 images batch.

- Comparison Table
## Augmented Datasets & Trained Models:
- Augmented Imagenette-160 datasets can be found at: https://drive.google.com/drive/folders/1EmhRXzn3hxRhxlJwDE1JH4H0WCOg1Xm_?usp=sharing.
- Trained Pytorch models can be found at: https://drive.google.com/drive/folders/1bHiVl3OqJ_cPFGhyv0Ph8cwYQmbFrnWG?usp=sharing.
## Reports
- Proposal: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/reports/Image%20Augmentations%20CNN%20Proposal.pdf.
- Final Research Paper:

## References
- C. Shorten and T. M. Khoshgoftaar, “A survey on Image Data Augmentation for Deep Learning,” Journal of Big Data, vol. 6, no. 1, 2019.
- K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” arXiv.org, 10-Dec-2015. [Online]. Available: https://arxiv.org/abs/1512.03385. [Accessed: 06-May-2021]. 
- Fastai, “fastai/imagenette,” GitHub. [Online]. Available: https://github.com/fastai/imagenette. [Accessed: 08-Apr-2021].
- Collections of Digital Image Processing Techniques: https://github.com/mnguyen0226/opencv-digital-image-processing
- One Cycle Learning Rate: https://github.com/dkumazaw/onecyclelr

## Honor Code @VT
You know what VT's Honor Code is, Hokies :). Don't do it. You have been warned.