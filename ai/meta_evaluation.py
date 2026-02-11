import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def run_meta_evaluation(dataset):

    results = []

    for row in dataset:
        prod_score = row['']