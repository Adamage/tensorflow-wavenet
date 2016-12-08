from __future__ import print_function
from __future__ import division

import os
import numpy as np


labels_pathname = './slt_arctic_demo_data/binary_label_425/'
normalized_labels_pathname = './slt_arctic_demo_data/binary_label_norm/'
include_entries_417_425 = True
epsilon = 0.01


def normalize_labels():

    labDir = os.listdir(labels_pathname)

    min_values = 10000000*np.ones(425, );
    max_values = -10000000*np.ones(425, );

    for label_filename in labDir:
        print(label_filename)
       
        labels = np.fromfile(labels_pathname + label_filename, dtype=np.float32);
        labels = labels.reshape((int(labels.shape[0]/425), 425))

        min_values = np.minimum(np.min(labels, axis=0), min_values)
        max_values = np.maximum(np.max(labels, axis=0), max_values)


    if not os.path.exists(normalized_labels_pathname):
        os.mkdir(normalized_labels_pathname)

    for label_filename in labDir:
        print(label_filename)

        labels = np.fromfile(labels_pathname + label_filename, dtype=np.float32);
        labels = labels.reshape((int(labels.shape[0]/425), 425))

        if include_entries_417_425:
            new_labels = np.zeros((labels.shape[0], 425), dtype=np.float32)
            num_entries = 425 
        else:   
            new_labels = np.zeros((labels.shape[0], 416), dtype=np.float32)
            num_entries = 416

        for i in range(num_entries):
            if (max_values[i] - min_values[i] > 0): 
                new_labels[:, i] = (1-2*epsilon)*(labels[:, i] - min_values[i])/(max_values[i] - min_values[i]) + epsilon;
            else:
                new_labels[:, i] = epsilon; # put an arbitrary value in interval [0, 1] 
                                            # Me may also remove that entry because it does not differntiate the full context phonemes    
        
        with open(normalized_labels_pathname + label_filename, 'wb') as fid:
            new_labels.tofile(fid)


if __name__ == '__main__':
    normalize_labels()

        


