from juliacall import Main as jl
jl.println("Hello from Julia!")

from juliacall import Pkg as jlPkg
jlPkg.activate(r"C:\Users\yasir\Desktop\brain_stuff\workshop2025")
jl.seval("using IntrinsicTimescales") # Evaluate the string: seval


import mne
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Download example data and specify the path
data_path = mne.datasets.brainstorm.bst_resting.data_path()
raw_fname = op.join(data_path, "MEG", "bst_resting", "subj002_spontaneous_20111102_01_AUX.ds")

# Read data
raw = mne.io.read_raw_ctf(raw_fname)

epochs = mne.make_fixed_length_epochs(raw, 10)
data = epochs.get_data()
fs = epochs.info["sfreq"]
print(data.shape)


acwtypes = jl.pyconvert(jl.Vector, [jl.Symbol(i) for i in ["acw50", "tau"]])
results = jl.acw(data, fs, acwtypes=acwtypes, dims=3, trial_dims=1, average_over_trials=True, parallel=True)

acw_results = results.acw_results
acw50 = np.array(acw_results[0])[0]
tau = np.array(acw_results[1])[0]

meg_indices = mne.pick_types(epochs.info, meg=True, ref_meg=False)

f, ax = plt.subplots(1, 2, figsize=(10,5))

mne.viz.plot_topomap(acw50[meg_indices], mne.pick_info(epochs.info, sel=meg_indices), axes=ax[0], show=False, cmap=mpl.cm.cool)
mne.viz.plot_topomap(tau[meg_indices], mne.pick_info(epochs.info, sel=meg_indices), axes=ax[1], show=False, cmap=mpl.cm.cool)

ax[0].set_title("ACW-50")
ax[1].set_title("Tau")

f.show()