import os

import numpy as np
import matplotlib.pyplot as plt


data_dir = './data/'

training_data = np.load(os.path.join(data_dir, 'training.npy'))
training_labels = np.load(os.path.join(data_dir, 'training_labels.npy'))
test_data = np.load(os.path.join(data_dir, 'test.npy'))


unique_labels, label_counts = np.unique(training_labels, return_counts=True)

print(f'Training data shape:\t{training_data.shape}')
print(f'Training labels shape:\t{training_labels.shape}')
print('Label counts:\n' + '\n'.join([f'{label}:{count: >7}' for label, count in zip(unique_labels, label_counts)]))

print(f'Test data shape_\t{test_data.shape}')


# np.random.seed(42)
n_samples = (2, 3) # (rows, colums)

fig = plt.figure(figsize=(2*n_samples[1], 1*len(unique_labels)*n_samples[0]))
figs = fig.subfigures(nrows=len(unique_labels))
for row, label in zip(figs, unique_labels):
    
    row.suptitle(f'Label: {label}', size=24)
    
    all_label_idxs = np.argwhere(training_labels == label).flatten()
    chosen_label_idxs = np.random.choice(all_label_idxs, np.prod(n_samples), replace=False)
    
    axs = row.subplots(*n_samples)
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    
    for chosen_label_idx, ax in zip(chosen_label_idxs, axs.flatten()):
        
        ax.set_title(f'Img index: {chosen_label_idx}')
        ax.imshow(training_data[chosen_label_idx])

plt.show()