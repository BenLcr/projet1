# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:06:45 2024

@author: guyadern
"""

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

pulse = np.loadtxt("pulse11.ASC")
data = scipy.io.loadmat('data.mat')
X1 = data["data1"]
j = 10
plt.plot(pulse[int(X1[j-1, 2]):int(X1[j-1, 3]), 1])
