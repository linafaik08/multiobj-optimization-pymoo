# Multi-Objective Optimization with NSGA-II & MCDM

- Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
- Creation date: April 2025  
- Last update: April 2025

## Objective

This repository contains code and experiments for solving **multi-objective dispatch problems** using **NSGA-II** and **MCDM (Multi-Criteria Decision Making)** tools. It aims to balance competing objectives like **maximizing sales** and **minimizing overstock** in a retail context under uncertainty.

For a deeper dive, check out the blog post in [The AI Practitionner](https://aipractitioner.substack.com/) page.

<div class="alert alert-block alert-info"> You can find all my technical blog posts <a href = https://linafaik.medium.com/>here</a>. </div>

## Project Description

This project simulates store-level inventory data, detects shortages, and applies evolutionary optimization to dispatch quantities. It then uses decision-making tools like ASF and pseudo-weights to recommend a final strategy among Pareto-optimal options.

### Data

The dataset used in this project is sourced from Kaggle: [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) by Anirudh Chauhan.

It contains inventory, sales, and forecasting data across multiple retail stores.  
In this repository, it is stored in:  `data/retail_store_inventory.csv`

### Code Structure

```
data/
├── retail_store_inventory.csv # Inventory data from Kaggle

notebooks/
├── multiobjective_dispatch_nsga2.ipynb # Main notebook for simulation, optimization & analysis

src/
├── optimization.py   # Dispatch problem and NSGA-II logic
├── mcdm.py           # Multi-Criteria Decision Making tools
├── visualization.py  # Plotting utilities

README.md
requirements.txt
```

## How to Use This Repository?

### Requirements

Install the dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Main libraries:
```
pymoo==0.6.1.3
plotly>=5.0.0
pandas>=1.3.0
numpy>=1.21.0
```

### Running the Project

1. Open `notebooks/multiobjective_dispatch_nsga2.ipynb`.
2. Follow the workflow:
   - Simulate shortages
   - Optimize dispatch with NSGA-II
   - Analyze trade-offs with MCDM
3. You can visualize and export results as needed using Plotly tools.
