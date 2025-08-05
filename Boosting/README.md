Installation

    1. Create the environment from the environment.yml file:    
        conda env create -f environment.yml

    2. Activate the environment:
        conda activate my_environment  # Replace "my_environment" with the actual environment name

Usage

    1. Ensure you are in the project's root directory for it to work properly

    2. Open the terminal and print:
        To train the model:
            python -m scripts.train_model # A model.pkl and processed_data.csv files will be created

        To evaluate the model
            python -m scripts.evaluate_model # You should get MSE, MAE and MAPE scores

Note

    I am new to this so if anything doesn't work or if you have any suggestions, please contact me via Telegram