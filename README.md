# 🏡 Housing & Rental Prediction Project

🔗 **Project Google Drive**:  
[Access the Drive Folder](https://drive.google.com/drive/folders/1wBVWORZl7w8UccK1tqoNAn0_LC3hptcO?usp=drive_link)

---

## Description

Welcome to My Group's Housing & Rental Forecasting Tool (For GA Tech CSE 6242).

This project delivers a comprehensive solution for predicting and visualizing real estate trends across U.S. ZIP codes. It integrates deep learning models and interactive dashboards to support both exploration and forecasting of housing and rental values.

### 🔧 Core Components
1. **Neural Network Model** – Predicts current and future housing/rental values using a Transformer-based architecture.  
2. **Interactive Visualization** – Dual dashboards for historical and forecasted data across ZIP codes.

### Key Features
- Forecasts up to 12 months into the future  
- Robust performance even with sparse data  
- Simple, modular training pipeline  
- Fully compatible with Google Colab and local Python (GPU recommended)

---

## Installation

1. Clone or download the repository.  
2. Open `redfin_model_pipeline.ipynb` — the primary notebook for running the project.  
3. Manually download the rental dataset and place it into the `data/` directory (instructions in the notebook).  
4. All other steps are automated via the notebook.

---

## Execution Steps

The notebook guides you through six main steps:

### Step 1: Initialize the Environment

- **Google Colab**: Run all cells under “Step 1. Initializing Environment.”
- **Local Setup**:
  - Skip Google Drive mounting.
  - In the second cell:
    - **Comment out**: `!pip install torch==...`
    - **Uncomment**: `!pip install -r requirements.txt`

### Step 2: Download and Prepare Data

- Run the first two cells to fetch and clean the housing dataset.
- Manually place rental data in `data/` as shown in the notebook.
- Run remaining preprocessing steps.

### Step 3: Impute Missing Values

- Use the provided interpolation function to handle missing entries.
- Recommended method: `"linear"`  
- Outputs are automatically saved as processed tensors.

### Step 4: Train the Model

The Transformer architecture is configured with the following:

- **Data Parameters**: `WINDOW_SIZE`, `PREDICTION_WINDOW`, `BATCH_SIZE`, etc.  
- **Model Parameters**: `EMBED_DIM`, `NUM_LAYERS_ENCODER`, `LR`, etc.  
- **Training Parameters**: `MAX_EPOCHS`, `EARLY_STOPPING_PATIENCE`, `CHECKPOINT_FILENAME`, etc.

Set `TRAIN_MODEL = True` to begin training. If `False`, the best pretrained model will be used.  
Metrics such as RMSE, MAE, R², and MAPE are displayed throughout training.

### Step 5: Test the Model

- Use a pretrained checkpoint to generate predictions.
- Visual outputs and saved forecasts are stored in `data/predictions/`.

### Step 6: Interactive Visualization

#### Flask Dashboard

Visualize ZIP-level trends and predictions:

- Run the first cell in this section to format predictions.
- Download required geoJSON files (as detailed in the notebook).

**Google Colab:**
- Run the Flask cell and click the generated URL (e.g., `https://localhost:5000/`).

**Local Setup:**
- Comment out Colab-specific lines:
  - `from google.colab...`
  - `output.serve_kernel...`
- Run Flask locally via `http://127.0.0.1:5000`

#### Tableau Dashboard

A pre-built Tableau dashboard enables:
- ZIP-level mortgage vs. rental cost comparison  
- Adjustable interest rates for mortgage calculations

Access it via the link titled **"Tableau Visualization"** in the notebook.

---

💡 *For questions, contributions, or issues — feel free to open an issue or submit a pull request!*
