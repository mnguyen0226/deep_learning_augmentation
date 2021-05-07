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
- Augmentation-method testing script can be found at: https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/test_image_aug.py.
    ```
    python test_image_aug.py
    ```

## How To Augment Image & Train Agents
- Note on set up machine for augmenting and training agents reprodution:
    - CUDA 10.0
    - Pytorch 1.2.0: https://pytorch.org/get-started/previous-versions/.
    - GPU that I use - Cloudera:
![alt text](https://github.com/mnguyen0226/image-augmentation-dnn-performance/blob/main/train_results/Cloudera.PNG)
    - Augmenting dataset(s) & training models will take a very long time, even with GPU. If you are interested in pre-augmented "Imagenette-160" and pre-trained ResNet9 models, please check the section below.

- Images Augmentation Task:
    - Step 1: Make sure your CPU is strong enough. I use 2 vCPU / 4 Gib Memory.
    - Step 2: Set up the right image path and image-save path in **process_image.py**.
    - Step 3: Choose the augmentation methods that you prefer in **process_image.py**.
    - Step 4 Run and Wait 10 minutes to 1.25 hour per sections of the training datasets (depends on what augmentation method you choose).
        ```
        python preprocess_image.py
        ```

- Training: 
    - Step 1: Make sure your CPU is strong enough. I use 2 vCPU / 4 Gib Memory with 1 GPU - CUDA v10.0.
    - Step 2: Choose the augmented dataset that you want in **model_train.py**.
    Step 3: Choose the path that you want to save your model in **model_train.py**.
    - Step 4 Run and Wait around 17 minutes per sections of the model training session.
        ```
        python model_train.py
        ```

## Results & Comparisons
- Results for training and validating ResNet9 on each image augmentation datasets can be found at: https://github.com/mnguyen0226/image-augmentation-dnn-performance/tree/main/train_results including:
    - "Accuracy vs Each Epoch" graphs.
    - "Learning Rate vs Batch Number" graphs.
    - "Loss Rate vs Each Epoch" graphs.
    - 1 testing image classification in testing batch.
    - Training results after 30 epochs.
    - Training shuffled 64 images batch.
    - Testing 64 images batch.

- Comparison Table of ResNet9's Performances on each dataset (the information is extracted from https://github.com/mnguyen0226/image-augmentation-dnn-performance/tree/main/train_results):

Dataset | Best Accuracy | Latest Accuracy at Epoch 30 | Latest Loss at Epoch 30 | Latest Learning Rate at Epoch 30 | Dataset Augmenting Time | ResNet9 Training Time on GPU
--- | --- | --- | --- | --- | --- | ---
Imagenette-160 Original | 72.09% | 72.09% | 0.01 | 0 | 0 | 17 minutes
Imageneete-160 Horizontal Flip | 78.53% | 62.24% | 0.01 | 0 | 45 minutes | 17 minutes
Imagenette-160 Vertical Flip | 77.25% | 73.54% | 0.01 | 0 | 45 minutes | 17 minutes
Imagenette-160 Random Rotation | 68.86% | 65.80% | 0.01 | 0 | 47 minutes | 17 minutes
Imagenette-160 Horizontal Shear | 75.26% | 72.16 | 0.01 | 0 | 47 minutes | 17 minutes
Imagenette-160 Vertical Shear | 74.41% | 73.41% | 0.01 | 0 | 45 minutes | 17 minutes
Imagenette-160 Translation | 74.31% | 63.22% | 0.01 | 0 | 48 minutes | 17 minutes
Imagenette-160 Ideal Highpass Filter | 68.69% | 66.64% | 0.01 | 0 | 185 minutes | 17 minutes
Imagenette-160 Ideal Lowpass Filter | 63.54% | 59.96% | 0.01 | 0 | 185 minutes | 17 minutes
Imagenette-160 Gaussian Highpass Filter | 70.42% | 66.88% | 0.01 | 0 | 185 minutes | 17 minutes
Imagenette-160 Gaussian Lowpass Filter | 69.15% | 63.71% | 0.01 | 0 | 185 minutes | 17 minutes
Imagenette-160 Butterworth Highpass Filter | 68.32% | 62.55% | 0.01 | 0 | 185 minutes | 17 minutes
Imagenette-160 Butterworth Lowpass Filter | 67.76% | 59.94% | 0.01 | 0 | 185 minute | 17 minutes
Imagenette-160 Adaptive Median Filter | 78.29% | 69.44% | 0.01 | 0 | 595 minutes | 17 minutes
Imagenette-160 Histogram Equalization | 69.4% | 68.30% | 0.01 | 0 | 46 minutes | 17 minutes
Imagenette-160 Invert | 69.80% | 67.62% | 0.01 | 0 | 43 minutes | 17 minutes
Imagenette-160 Canny Edge Detection | 62.63% | 61.99% | 0.01 | 0 | 130 minutes | 17 minutes

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
- One Cycle Learning Rate: https://github.com/dkumazaw/onecyclelr.

## Honor Code @VT
You know what VT's Honor Code is, Hokies :). Don't do it. You have been warned.