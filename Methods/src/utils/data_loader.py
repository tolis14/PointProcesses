import torch
from pandas import read_csv
from src.synth_data import PoissonProcess

def load_synth_data(intensity: callable, max_time: torch.float, bound: torch.float):
    pp = PoissonProcess(intensity, max_time, bound)
    pp.simulate()
    data = pp.get_data()
    return data

def load_real_data(dataset_name: str):
    path = '../data/' + dataset_name + '.csv'
    df = read_csv(path)
    data = torch.tensor(df.values, dtype=torch.float)
    return data