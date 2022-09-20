# Building Shazam From Scratch
![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

In this repository we tried to implement a simplified copy of the **Shazam** application able to tell you the name of a song listening to a short sample.

## Overview

1. Converting the songs from mp3 to wav with Librosa and extraction of the peaks.
2. MinHashing with permutations on the shingles matrix.
3. Locality sensitive hashing to divide the songs in buckets.
4. Shazam!

## Contents

- _pickle_ is a folder that contains the songs peaks, the shingles array and the shingle matrix in pickle format.
- _ShazamLSH.ipynb_ is the main notebook that only contains the explanation of the steps and some comments.
- _function.py_ contains all the implemented function needed to execute the notebook.

## Resources
This is the dataset we used and processed:
- https://www.kaggle.com/dhrumil140396/mp3s32k

We also share some useful links can help to understand what is the process behind Min Hashing and LSH in order to recognise song:
- https://willdrevo.com/fingerprinting-and-audio-recognition-with-python/
- https://www.learndatasci.com/tutorials/building-recommendation-engine-locality-sensitive-hashing-lsh-python/
- https://librosa.org/
