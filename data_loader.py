import numpy as np
import pandas as pd

def load_dataset():
    train_dataset = pd.read_csv(r"./facial_expressions/image_emotion.csv")
    return train_dataset