import pickle

# Function to load the pickle file
def load_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)