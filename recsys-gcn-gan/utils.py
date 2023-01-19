import pandas as pd

def save_as_picke(data, path) -> None:
    data.to_pickle(path)