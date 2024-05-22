# Decentralized Optimization with Coupled Constraints

This repository contains the code for the experiments related to our paper titled "Decentralized Optimization with Coupled Constraints", which we have submitted to the 38th Conference on Neural Information Processing Systems (NeurIPS 2024).

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Files Description](#files-description)

## Installation <a name="installation"></a>
To use this project, you need to have Python and Jupyter Notebook installed on your computer. You can download Python from [here](https://www.python.org/downloads/) and Jupyter Notebook from [here](https://jupyter.org/install).

Clone the repository:
```bash
git clone https://github.com/kisnikser/Decentralized-Optimization.git
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage <a name="usage"></a>
To run the project, open the Jupyter Notebook:
```bash
jupyter notebook
```
Then, open the file you are interested in using the Jupyter Notebook interface.

## Files Description <a name="files-description"></a>
In the root directory, you will find the following files:
- `utils.py`: supplementary functions.
- `models.py`: implementations of the problem models that we have used to test various algorithms. The most important thing that these models should return is the problem parameters, as well as the oracles for the function, gradient, and hessian.
- `methods.py`: implementations of our algorithms, as well as the Tracking-ADMM and DPMM algorithms that we compare against.
- `example.ipynb`: results obtained on the first problem from the experiments section.
- `vfl.ipynb`: results obtained on the Vertical Federated Learning (VFL) problem.

Additionally, there is a directory named `data` which contains text files for uploading datasets `mushrooms`, `a9a`, and `w8a`.