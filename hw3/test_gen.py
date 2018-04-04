import numpy as np
import matplotlib.pyplot as plt
image = np.load('trainX.npy').reshape(-1,48,48)
plt.imshow(image[777],cmap='gray')
plt.show()