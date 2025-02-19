import os
import sys
import re
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Union, List

# stop_list = ['', '\n']

# def read_float(filename, split = " "):
#     filename = glob.glob(filename)[0]
#     assert isinstance(filename, str)
#     assert isinstance(split, str)
#     return_list = []
#     with open(filename, 'r') as f:
#         tmpstr = f.readline()
#         while(tmpstr):
#             tmplist = re.split(split, tmpstr)
#             reslist = []
#             for i in stop_list:
#                 try:
#                     tmplist.remove(i)
#                 except:
#                     pass
#             for i in tmplist:
#                 tmp = float(i)
#                 # assert tmp > 0
#                 reslist.append(tmp)
#             return_list.append(reslist)
#             tmpstr = f.readline()
#     return return_list
def read_float(filename: str) -> np.array:
    filename = glob.glob(filename)[0]

    data: List[float]=[]
    with open(filename, 'r') as f:
        tmpstr = f.readline()
        while (tmpstr):
            data.append(float(tmpstr))
            tmpstr = f.readline()
    return np.array(data)

def import_from_logs(program: str, topic: str, dataset: str, accuracy: float) -> np.array:
    data=[]
    for k in ["10", "100", "1000"]:
        data.append(read_float(f"{program}-{topic}-{dataset}_{k}-{k}-{accuracy}*.log"))
    return np.array(data)

conann_fnrs = import_from_logs("ConANN", "error", "bert", "0.2")
faiss_fnrs = import_from_logs("Faiss", "error", "bert", "0.2")

print(conann_fnrs.shape)
print(faiss_fnrs.shape)

fig, axs = plt.subplots(2, 3, figsize=(16, 9))

for i in range(3):
    axs[0, i].hist(conann_fnrs[i], bins=50)
    axs[0, i].set_title(f'ConANN k={10**(i+1)}')
    axs[0, i].set_xlabel('FNR')
    axs[0, i].set_ylabel('Frequency')

    axs[1, i].hist(faiss_fnrs[i], bins=50)
    axs[1, i].set_title(f'Faiss k={10**(i+1)}')
    axs[1, i].set_xlabel('FNR')
    axs[1, i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

conann_cls = import_from_logs("ConANN", "efficiency", "bert", "0.2")
conann_avgs = np.mean(conann_cls, axis=1)
print(conann_avgs)
faiss_cls = import_from_logs("Faiss", "efficiency", "bert", "0.2")

print(conann_cls.shape)
print(faiss_cls.shape)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i in range(3):
    axs[i].hist(conann_cls[i], bins=50)
    axs[i].axvline(x=faiss_cls[i], color='r', linestyle='--', label='Faiss')
    axs[i].axvline(x=conann_avgs[i], color='b', linestyle='--', label='ConANN average')
    axs[i].set_title(f'k={10**(i+1)}')
    axs[i].set_xlabel('Clusters')
    axs[i].set_ylabel('Frequency')
    axs[i].legend()

plt.tight_layout()
plt.show()
