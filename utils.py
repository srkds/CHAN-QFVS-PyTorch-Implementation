import pickle
import json

def load_json(filename):
    with open(filename, encoding='utf8') as f:
        return json.load(f)

# Function to load the pickle file
def load_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)