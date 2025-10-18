import numpy as np
import os
from utils import *
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from fun import *


if __name__ == '__main__':
    ## load the data
    path = os.getcwd() + '/../data/'  
    images_np = images_np_load(path, plot=False)
    #images_plot(images_np)

    offsets = log_polar_offsets(41)

    images_self_sim = [0] * 5
    self_sim_polarlog = [0] * 5

    images_self_sim[0] = self_loc_sim_get(np.transpose(images_np[0], [1, 2, 0]))
    i = images_self_sim[0][100][100]
    r, theta = log_polar_conversion(i)
    log_polar_map = update_log_polar_map(i, r, theta)

    visualize_logpolar_feature(log_polar_map, r)
    visualize_similarity_map(i)