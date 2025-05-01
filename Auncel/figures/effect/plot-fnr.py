import os
import sys
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import palettable
import glob

stop_list = ['', '\n']

def read_float(filename, split = " "):
    assert isinstance(filename, str)
    assert isinstance(split, str)
    print(filename)
    filename = glob.glob(filename)[0]
    return_list = []
    with open(filename, 'r') as f:
        tmpstr = f.readline()
        while(tmpstr):
            tmplist = re.split(split, tmpstr)
            reslist = []
            for i in stop_list:
                try:
                    tmplist.remove(i)
                except:
                    pass
            for i in tmplist:
                tmp = float(i)
                # assert tmp > 0
                reslist.append(tmp)
            return_list.append(reslist)
            tmpstr = f.readline()
    return return_list

def read_parameter():
    if len(sys.argv) != 2:
        print("Usage: python plot-fnr.py <parameter>")
        sys.exit(1)
    parameter = sys.argv[1]
    return parameter

dataset = read_parameter()


font_size = 30
plt.rc('font',**{'size': font_size, 'family': 'Arial'})
plt.rc('pdf',fonttype = 42)
fig_size = (9, 6)
fig, axes = plt.subplots(figsize=fig_size)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=None)

file_base = f"../../../results/Auncel/logs/Auncel-error-{dataset}_100-100"
accs = ["0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]
display_accs = [int(float(i)*100) for i in accs]
data = {i: [] for i in display_accs}
for i in accs:
    res = read_float(f"{file_base}-{i}-*.log")
    for j in res:
        data[int(float(i)*100)].append(int(100 - (1-j[0])*100))
# for i in data:
#     print(len(data[i]), sum(data[i])/len(data[i]), min(data[i]), max(data[i]))
x = [i for i in data]
for i in data:
    data[i].sort()
y_tail95 = [data[i][int(0.95*len(data[i]))] for i in data]
y_worst = [max(data[i]) for i in data]

# Plot bars
# top
l1 = axes.plot(x, x, label='Ideal', marker=None, color='k', lw=2.4, zorder=4)
l3 = axes.plot(x, y_worst, label='Maximum Error', marker='^', color='r', lw=3.6, zorder=3, markersize=11)
l2 = axes.plot(x, y_tail95, label='95%-tile Error', marker='o', color='royalblue', lw=3.6, zorder=3, markersize=11)

axes.set_xlim(left=10, right=70)
axes.set_ylim(bottom=0, top=100)

axes.set_xlabel('Requested Error Bound (%)')
axes.set_ylabel('Actual Error (%)')
x_ticks = [i for i in data]
axes.set_xticks(x_ticks)
axes.grid(axis='y', linestyle='--')
# ax1.get_yaxis().set_tick_params(pad=12)
# axes.legend(frameon=False, ncol=1, loc='upper center', bbox_to_anchor=(0.3, 1.0), prop={'size': font_size})
axes.set_title(dataset)

# Save the figure
# file_path = '13-1.pdf'
# plt.savefig(file_path, bbox_inches='tight', transparent=True, backend='pgf')
# CHANGED TO
plt.savefig(dataset, bbox_inches='tight', transparent=False)