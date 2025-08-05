# project_config.py
import os
import sys

# Set the root directory (one level up from the current file's directory)
ROOT_DIR = os.path.dirname(__file__)

# Add ROOT_DIR to sys.path
sys.path.append(ROOT_DIR)

# Change the current working directory to ROOT_DIR (optional, for data access)
os.chdir(ROOT_DIR)