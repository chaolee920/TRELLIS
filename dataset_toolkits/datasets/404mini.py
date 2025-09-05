import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd


def get_metadata(**kwargs):
    metadata = pd.read_csv("/workspace/proj-sub17/TRELLIS/datasets/404mini/metadata.csv")
    return metadata