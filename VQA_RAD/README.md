# Visual Question Answering (VQA) on Radiology-VQA
## Overview
This course project for computer vision-CSCI-GA.2271-001 involves the development of a Visual Question Answering (VQA) system, employing a multimodal architecture. This system is designed to accept both an image and a question as input and generate an accurate answer as output. Our primary objective is to utilize the image in conjunction with a CNN and an LSTM to correctly respond to the posed question. We leverage the Radiology VQA dataset for this project.

## Problem : Question Answering for the Medical Images 
- [Data](https://www.nature.com/articles/sdata2018251)
- [Stacked Attention Network](https://arxiv.org/pdf/1511.02274.pdf)


## Dataset
The dataset used in this project comprises a variety of radiological images paired with naturally occurring questions and answers generated and validated by clinicians. The dataset includes 315 radiological images, encompassing different categories like head, chest, and abdomen. A total of 3,515 visual questions are part of the dataset, with a mix of free-form, rephrased, and framed questions. This dataset aims to facilitate the development of VQA tools in the medical domain, particularly for applications in radiology.

## Running the code
We have compiler the training and testing code in iPython Notebook to simplify the setup process

1. Launch the rad_vqa.ipynb.
2. Execute all the code cells. 
3. Proceed to the inference section that comes after the training process.
4. Feel free to modify the 'question' variable to ask any question you desire.

Please download the following files and folders:
#### Download : 

        1. trainset.json
        2. testset.json
        3. VQA Image Folder
        4. Cache Folder  (contains the pickle file, for converting the answers to labels, and vice versa, and the mapping for dictionary and answer)
Once you have obtained these resources, please upload them to your Google Drive. Afterward, follow the instructions provided in the VQA.ipynb notebook to continue with the project.



