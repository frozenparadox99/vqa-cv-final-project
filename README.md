# VQA-CV Final Project

## Table of Contents
- [VQA-CV Final Project](#vqa-cv-final-project)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Create the virtual environment](#create-the-virtual-environment)
    - [Clone the repository and install the dependencies](#clone-the-repository-and-install-the-dependencies)
    - [Installing the dataset](#installing-the-dataset)
  - [Plan of Action](#plan-of-action)

## Installation

### Create the virtual environment
```bash
python -m venv vqa
source vqa/bin/activate
```

### Clone the repository and install the dependencies
If you are not able to install the requirements file, just install all dependencies one after the other and remove everything other than the -e . from the requirements file
```bash
git clone https://github.com/yourusername/vqa-cv-final-project.git
cd vqa-cv-final-project
pip install -r requirements.txt
```

### Installing the dataset
```bash
python3 main.py
```

## Plan of Action

- [ ] Create a data ingestion script which can download, unzip and pre-process the data.
- [ ] Create a "prepare base model" script which would contain a sample attempt at VQA using the current baseline.
- [ ] Create a training script.
- [ ] Create an evaluation script.
- [ ] Once the baseline has been implemented, research in order to get a better working model.