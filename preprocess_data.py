import os
import pickle
import numpy as np
path = 'data'
save_path = 'clean_data'
for file in os.listdir(path):
    full_path = os.path.join(path, file)
    with open(full_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    actor = str(int(file[1:3]))
    for i in range(data['data'].shape[0]):
        video = str(i)
        save_name = os.path.join(save_path, "data_" + actor + "_" + video + ".npy")
        label_name = os.path.join(save_path, "label_" + actor + "_" + video + ".npy")
        np.save(open(save_name, 'wb'), data['data'][i])
        np.save(open(label_name, 'wb'), data['labels'][i])