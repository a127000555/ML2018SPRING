import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
Y = np.array([row for row in open('train_Y','r')]).astype(np.float).reshape(-1,1)
df = pandas.read_csv('train_X')
print(df['marital_status_Married-civ-spouse'])
print(df.columns.values)
