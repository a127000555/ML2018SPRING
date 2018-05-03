import sys
import numpy as np 
from skimage import io
def reconstruct(M):
	M -= np.min(M)
	M /= np.max(M)
	M = (M*255).astype(np.uint8)
	return M

data = []
N = 415
for idx in range(N):
	ig = io.imread( sys.argv[1] + '/' + str(idx) + '.jpg')
	data.append(np.array(ig))

k = 415
data = np.array(data)
X = (data.reshape(N,-1)[:k]).T
X_mean = np.mean(X,1).reshape(-1,1)
print('start svd')
U , s , V = np.linalg.svd(X - X_mean , full_matrices=False)

target_idx = int(sys.argv[2][:-4])

target = data[target_idx].reshape(-1) - X_mean.reshape(-1)

dot_value = np.dot(target,U[:,:4])
print((dot_value*U[:,:4]).shape)
re = np.sum(dot_value*U[:,:4],1) + X_mean.reshape(-1)
io.imsave('reconstruction.png',reconstruct(re).reshape(600,600,3))
