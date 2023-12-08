import pandas as pd

dial_label = {
    0 : 'AST',
    1 : 'EU',
    2 : 'GL',
    3 : 'AN',
    4 : 'LAD'
}

fold_label = {
    'AST' : 0,
    'EU' : 1,
    'GL' : 2,
    'AN' : 3,
    'LAD' : 4
}

human_readable_label = {
    0 : 'Asturian',
    1 : 'Neapolitan',
    2 : 'Galician',
    3 : 'Aragonian',
    4 : 'Judeo-Latin'
}

def explain_label(label : int) -> str:
    """ 
    Given an integer label, convert it to the corresponding string label

    :param int label: integer label to be converted
    :return: string corresponding to the given label

    """
    return dial_label[label]

def encode_label(label : str) -> int:
    """ 
    Given a string label, encode it to the corresponding index

    :param string label: string label to be converted
    :return: int corresponding to the given label

    """
    return fold_label[label]

def human_readable_label(label) -> str:
    """
    Given a label, produce a human readable label for the corresponding dialect.
    """
    if isinstance(label, str): label = fold_label(label)
    return human_readable_label[label]



def load_data(train_path = "train.csv"):
    """
    Load training and evaluation data.
    """
    if train_path is not None:
        data_train = pd.read_csv("train.csv")
        X_train, y_train = data_train['text'].values, data_train['label'].values.astype(int)
        return X_train, y_train
        
    return X_train, y_train