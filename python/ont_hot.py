
# date: 20181012
# chunhua liu
# function: use np.eye to generate one-hot vector
# realization: use np.eye to generate M*N matrix, the following array indicate the offset 0f 1

import numpy as np


a =[[0.7,0.1,0.2],[0.2,0.3,0.5],[0.1,0.6,0.3]]

b =np.argmax(a,axis=1)

n_values = np.max(b) + 1
n_eye = np.eye(n_values)[b]
print (n_eye)


# conver to bool
n_boolean = np.bool_(n_eye)
print (n_boolean)
