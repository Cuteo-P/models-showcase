
# Car Price Prediction

**A simple gradient boosting(catboost) model to predict the car prices**

---

##  Installation

1. **Create the environment from the `environment.yml` file**:  
   Run the following command in your terminal:
   ```bash
   conda env create -f environment.yml --name your_env_name # Replace "your_env_name" with a new name
   ```

2. **Activate the environment**:  
   Activate your environment using the command:
   ```bash
   conda activate your_env_name  # Replace "your_env_name" with the name you created in the previous step
   ```

---

##  Usage

### 1. Ensure you are in the project's root directory

Make sure to run the following commands from the root directory of the project to ensure everything works properly.

### 2. Running the Model

#### To **train the model**:
Run the command below:
```bash
python -m scripts.train_model.py
```
This will generate two files: 
- `model.pkl` (the trained model)
- `processed_data.csv` (the processed data)

#### To **evaluate the model**:
Run the following command:
```bash
python -m scripts.evaluate_model.py
```
You should get the following evaluation metrics:
- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**

---

## ⚠️ Note

- **I am new to this**: If anything doesn’t work, or if you have any suggestions or improvements, please feel free to contact me via Telegram.

---

##  Contact

- **Telegram**: [https://t.me/frozenfoxby]