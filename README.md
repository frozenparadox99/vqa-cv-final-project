# VQA-CV Final Project

## Table of Contents
- [VQA-CV Final Project](#vqa-cv-final-project)
  - [Table of Contents](#table-of-contents)
- [About the Project:](#about-the-project)
  - [Problem : Visual Question Answering for the Medical Images](#problem--visual-question-answering-for-the-medical-images)
  - [Setup:](#setup)
      - [Download](#download)
      - [Important Links](#important-links)
  - [Pre Processing](#pre-processing)
  - [Model Description](#model-description)
  - [Training and Testing](#training-and-testing)

# About the Project:

## Problem : Visual Question Answering for the Medical Images 
- [Data](https://www.nature.com/articles/sdata2018251)
- [Stacked Attention Network](https://arxiv.org/pdf/1511.02274.pdf)
- [Slides for Stacked Attention Network](http://www.cs.virginia.edu/~vicente/vislang/slides/wasimonica.pdf)

## Setup: 


#### Download 

        1. trainset.json
        2. testset.json
        3. VQA Image Folder
        4. Cache Folder 
Upload these to your google drive, and then follow the instructions that are present in the radiology_vqa_notebook.ipynb.


#### Important Links 
- [Google Colab: radiology_vqa_notebook.ipynb](https://colab.research.google.com/drive/1Bss0nh2MkNj0a2gAJUjZwDdOVU-dbeDP?usp=sharing)

- [Google News Vector](https://drive.google.com/file/d/1ppatycyNv1UtYrlhHU3jo6ZuuY3SPCeL/view?usp=sharing)

- [Demo Video 1](https://drive.google.com/file/d/1WXSk8TfH6QydYYUkzXyUOhQYEcThpAyI/view?usp=sharing)

- [Demo Video 2](https://drive.google.com/file/d/18T7CG96it7DLMz-QpFPqn33aC1PwT6By/view?usp=sharing)

## Pre Processing
All the pre-processing steps are highlighted in the following PDF:
- [Pre Processing.pdf](https://drive.google.com/file/d/1_PGB9WQylvzkT1eMfmyqURLxUh5jHgaj/view?usp=sharing)

## Model Description
![img](https://i.imgur.com/PdzxNgb.png)
![img6](https://i.imgur.com/QsRitxy.png)

## Training and Testing 
### Preparing the train test split and model
![img3](https://i.imgur.com/BsR3Nrd.png)
### Accuracy on the test set
![img4](https://i.imgur.com/dsxcFAs.png)
### Accuracy vs Epoch Plot
![img2](https://i.imgur.com/nR95v7e.png)
