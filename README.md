# BUIR_ReChorus

## Installation

1. Clone the project

```bash
git clone 
```

2. Install dependencies

Each model has its own dependency file. Enter the corresponding model directory and run the following command.  
**Note:** The dependencies for the `BUIR_ReChorus` model are different from those of the other models.

```bash
pip install -r requirements.txt
```

## Model Overview

### 1. BUIR Model

Source: https://github.com/donalee/BUIR

**BUIR: Bootstrapping User and Item Representations for One-Class Collaborative Filtering**

The BUIR model helps understand and optimize the interaction process between users and the system by analyzing four core elements: **Behavior**, **User**, **Information**, and **Result**. It focuses on user needs and behaviors, ensures information is conveyed effectively, and ultimately improves user experience and task completion performance.

### 2. BUIR_ReChorus Model

We reproduced the **BUIR_NB** model using the **ReChorus** framework. It is located at:

- `BUIR_ReChorus/src/models/general/BUIR_NB`

ReChorus is a deep-learning-based recommender system framework.

### 3. BUIR_PYG Model

We improved the original BUIR model. In the original **BUIR_NB**, graph convolution was implemented with custom code; we replaced it with the **PyG** library to implement graph convolution, and found that it performed better than the original implementation.

This model is located at:

- `BUIR_PYG/Models/BUIR_NB`

Run with:

```bash
python main.py --dataset toy-dataset --model buir-nb
```

> Note: Since the original authors did not have GPUs, **all code runs on CPU**.

### 4. BUIR_GAT Model

Based on the original **BUIR_NB** model, we further introduced an **attention mechanism (GAT)** to improve the model.

This model is located at:

- `BUIR_GAT/Models/BUIR_NB`

Run with:

```bash
python main.py --dataset toy-dataset --model buir-nb
```

### 5. BUIR_GCN Model

Based on the original **BUIR_NB** model, we also introduced **interpretability** for graph neural networks and improved the model using **GCN**.

This model is located at:

- `BUIR_GAT/Models/BUIR_NB`

Run with:

```bash
python main.py --dataset toy-dataset --model buir-nb
```

## Experiments

### 1. Reproduction Experiments

We reproduced the **BUIR_NB** model in the **ReChorus** framework, and used three datasets from ReChorus—**Grocery_and_Gourmet_Food** and **MIND_Large**—to compare against two other models of the same category in the original ReChorus framework (**BRRMF**, **NEUMF**).

### 2. Improvement Experiments

We improved the original **BUIR_NB** model in the following ways:

- Using the **PyG** library to implement graph convolution
- Introducing an **attention mechanism (GAT)**
- Introducing **interpretability** for graph neural networks

We then conducted comparative experiments on the original **BUIR_NB** dataset: **toy-dataset**.
