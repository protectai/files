import pickle
import random

random.seed(42)


def load_captions(sample_size=1000):
    """
    Load training and validation key-value pair data for captions.  
    
    """    
    train_dataset = None
    valid_dataset = None
    
    with open('train_dataset.pickle', 'rb') as handle:
        train_dataset = pickle.load(handle)
       
    with open('validation_dataset.pickle', 'rb') as handle:
        valid_dataset = pickle.load(handle)
    
    if sample_size > 0:
        random_keys = random.sample(train_dataset.keys(), sample_size)
        train_dataset = {key: train_dataset[key] for key in random_keys}
        
    return train_dataset, valid_dataset

    