# Convolutional Hierarchical Attention Network for Query-Focused Video Summarization (CHAN): A PyTorch Implementation

This is a PyTorch implementation of the [**"Convolutional Hierarchical Attention Network for Query-Focused Video Summarization"**](https://arxiv.org/abs/2002.03740), which is accepted by AAAI 2020 conference.

> Note: This project is stil a work in progress

## 🎥 Model Details

## 📑 Dataset

## 📈 Loss Function and Evaluation Method

## 📊 Results

Here is the result video summary for the query `FOOD` and `HANDS`. The model **generated a ~4:30 minute summary which contains clips that either have food or hands in frame from a ~4-hour long video** which contains diverse scenes like library, mall, driving, shop, etc.

https://github.com/srkds/CHAN-QFVS-PyTorch-Implementation/assets/61644078/5ed127f7-06fe-4d91-85e7-9626ebc38b6c

## Installation

### Step 1: Install dependencies

```py
pip install -r requirements.txt
```

### Step 2: Run the model

```py
python main.py
```

## Model Settings and Experiment Details

## Todo

- [ ] Add self attention and query focused global attention.

## 🙏 Acknowledgement

The implementation and understanding of this paper is being done as part of my research progress under the guidance of [Prof. Payal Prajapati](https://ldce.ac.in/faculty/payal.prajapati.129).

The evaluation code is being borrowed from EgoVLPv2.

The code is inspired by CHAN implementation:
https://github.com/ckczzj/CHAN
