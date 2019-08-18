import subprocess
from StringIO import StringIO
import pandas as pd
import os
import torch
import json
def save(model, niter, save_folder):
  if not os.path.exists(save_folder):
    os.makedirs(save_folder)
  torch.save(model, os.path.join(save_folder, "model" + str(niter) + ".pt"))

def save_config(config, save_folder):
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, "conf.json")
  with open(config_file, "w") as f:
    f.write(json.dumps(dict(config.__dict__)))

def save_config(config, save_folder):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    config_file = os.path.join(save_folder, "conf-pytorch.json")
    with open(config_file, "w") as f:
        f.write(json.dumps(dict(config.__dict__)))

def isnan(x):
    return x != x
    
def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    idx = gpu_df['memory.free'].astype(float).idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx