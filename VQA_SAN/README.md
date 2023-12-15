# Visual Question Answering (VQA) using Stacked Attention Networks (SAN)
## Overview
This course project for computer vision-CSCI-GA.2271-001 project involves the development of a Visual Question Answering (VQA) system, employing a multimodal architecture. This system is designed to accept both an image and a question as input and generate an accurate answer as output. Our primary objective is to utilize the image in conjunction with a CNN and an LSTM to correctly respond to the posed question.

## Baseline Approach
In our initial approach, we created a 2-channel model combining visual and language inputs by performing point-wise multiplication on their embeddings. We subsequently passed these combined embeddings through fully-connected layers. Our objective was to derive a probability distribution encompassing the various potential answer classes to arrive at the final answer.

Architecture:
![image](https://user-images.githubusercontent.com/38180831/213108042-741bfe93-63de-4b9b-a958-0e2e3c489e74.png)

The different components of architecture are outlined as follows:

- **Image Channel**
  - Provides 1024-dimensional image embeddings.
  - Utilizes VGGNet activations and a fully connected layer with tanh non-linearity.

- **Question Channel**
  - Generates 2048-dimensional question embeddings.
  - Employs a two-layer LSTM, concatenating the last cell state and hidden state.
  - Transforms the embeddings to 1024 dimensions using a fully connected layer with tanh non-linearity.

- **Multi-Layer Perceptron (MLP)**
  - Combines image and question embeddings via point-wise multiplication.
  - Comprises two linear layers, each with 1000 hidden units.
  - Includes dropout (0.5), tanh activation, and a softmax layer.
  - Outputs a probability distribution over a 1000-word answer vocabulary.

## Stacked Attention Network Approach
Architecture:
![image](https://user-images.githubusercontent.com/38180831/213108602-17595b09-fc7d-44ba-a687-faaf6943a1ed.png)

To boost VQA system performance, we integrated stacked attention layers into the architecture, resulting in the advanced DL approach with a 54.82% accuracy. This approach consistently delivered sensible answers in its top-5 predictions by effectively leveraging multi-step reasoning and focusing on relevant image regions.

## Dataset
The project employs the most recent version of the standard Visual Question Answering dataset, known as Visual VQA dataset v2.0. This dataset includes 82,783 MS COCO training images, 40,504 MS COCO validation images, and 81,434 MS COCO testing images. Additionally, it contains 443,757 questions for training, 214,354 questions for validation, and 447,793 questions for testing. In total, there are 4,437,570 answers for training and 2,143,540 answers for validation within this dataset.

To access the dataset, please download and extract it from the official VQA website at the following URL: https://visualqa.org/download.html.

## Running the code
We have compiler the training and testing code in iPython Notebook to simplify the setup process

1. Launch the Baseline_Network.ipynb or SAN_Network.ipynb file.
2. Execute all the code cells. Keep in mind that the training section may throw an error since no images are loaded, but it's crucial to run it to set up the models.
3. Proceed to the inference section that comes after the training process.
4. Feel free to modify the 'question' variable to ask any question you desire.

## Results
The stacked attention network outperformed the baseline, producing an accuracy of 54.82%. The stacked attention layers helped in multi-step reasoning and focusing on relevant portions of the image to detect the answer.

## References
* Paper implementation
  + Paper: VQA: Visual Question Answering
  + URL: https://arxiv.org/pdf/1505.00468.pdf
    
