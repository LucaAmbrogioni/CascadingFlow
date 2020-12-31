import pickle
import os
from scipy.stats import sem
import numpy as np
from shutil import copyfile

model_name = "vol"
lik_name = "c"
d_x = 2
exp_name = model_name + "_" + lik_name

results = {}


pickle_in = open(f"{exp_name}_results/uni_results.pickle", "rb")
results['uni'] = pickle.load(pickle_in)

pickle_in = open(f"{exp_name}_results/multi_results.pickle", "rb")
results['multi'] = pickle.load(pickle_in)

pickle_in = open(f"{exp_name}_results/pred_results.pickle", "rb")
results['pred'] = pickle.load(pickle_in)

for k, models in results.items():
    print(f'{k}')
    s = ''
    for m, values in models.items():
        s+=f'{m}'
        for v in values:
            s+=f' & {v:.3f}'
        s += f"\\\\\n"

    print(s)

for k, models in results.items():
    print(f'{k}')
    s = ''
    for m, values in models.items():
        s+=f'{m}'
        s += f' & {np.mean(values):.3f} $\\pm$ {sem(values):.3f}'
        s += f"\\\\\n"

    print(s)

if not os.path.isdir(f'{exp_name}_selected_figures'):
    os.makedirs(f'{exp_name}_selected_figures')
for m, values in results['uni'].items():
    max_v = np.argmax(values)
    for r in range(d_x):
        file_name=f'{m}_rep:{max_v}_{r}.png'
        copyfile(f'{exp_name}_figures/{file_name}', f'{exp_name}_selected_figures/{file_name}')



