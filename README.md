# Setup

#### 1. Enter the project directory

```bash
cd path/to/project
```

#### 2. Create a Python virtual environment

```bash
python -m venv .venv
```

#### 3. Activate the virtual environment

- **Mac/Linux:**
    ```bash
    source .venv/bin/activate
    ```
- **Windows Powershell:**
    ```bash
    .venv\Scripts\activate.ps1
    ```

- **Windows Command Prompt:**
    ```bash
    .venv\Scripts\activate
    ```

#### 4. Install required packages

```bash
python -m pip install -r requirements.txt
```

#### 5. Exit the virtual environment

```bash
deactivate
```

# Usage

### All script must be run within the virtual environment created above.

To do so you can either reactivate the virtual environment (step 3 above) or use an inline method where you pass the script name to the Python interpreter within the virtual environment, like so:

- **Mac/Linux:**
    ```bash
    .venv/bin/python script_name.py
    ```
    
- **Windows:**
    ```bash
    .venv/Scripts/python.exe script_name.py
    ```

### Here are the main scripts you can run (in order of typical workflow):

1. `import_dataset.py` - Imports and preprocess data. Takes a command line argument for the path to the external dataset. Expects a specific data format and may need to be adjusted based on your dataset.
```bash
import_dataset.py path/to/dataset/
```

2. `train.py` - Trains the model using the preprocessed data. Hyperparameters can be adjusted within `config.yaml`. Also tests the model after training and saves the trained model and test data to /runs/detect/...
```bash
train.py
```

3. `export_model.py` - Exports the trained model to format defined in `config.yaml` for deployment. Asks the user to select a training run to use `best.pt` from.
```bash
export_model.py
```

4. `inference.py` - Runs inference using the `final_model` in format defined in `config.yaml`. Expects the model to be in the base directory. Takes a command line argument for the path to the image folder for inference. Outputs results to runs/predict/...
```bash
inference.py path/to/image_folder/
```