# %matplotlib inline
# 主成分分析法对数据进行降维。采用方式是svd方法。GitHub上大神的demo地址是https://github.com/eliorc/Medium/blob/master/PCA-tSNE-AE.ipynb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from tensorflow_core.examples.tutorials.mnist import input_data


class TF_PCA:
    def __init__(self, data, dtype=tf.float32):
        self._data = data
        self._dtype = dtype
        self._graph = None
        self._X = None
        self._u = None
        self._singular_values = None
        self._sigma = None

    def fit(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._X = tf.placeholder(self._dtype, shape=self._data.shape)
            singular_values, u, _ = tf.svd(self._X)
            sigma = tf.diag(singular_values)
        with tf.Session(graph=self._graph) as sess:
            self._u, self._singular_values, self._sigma = sess.run(
                [u, singular_values, sigma], feed_dict={self._X: self._data})

    def reduce(self, n_dimensions=None, keep_info=None):
        if keep_info:
            normalized_singular_values = self._singular_values / sum(self._singular_values)
            info = np.cumsum(normalized_singular_values)
            index = next(idx for idx, value in enumerate(info) if value >= keep_info) + 1
            n_dimensions = index
        with self._graph.as_default():
            sigma = tf.slice(self._sigma, [0, 0], [self._data.shape[1], n_dimensions])
            pca = tf.matmul(self._u, sigma)

        with tf.Session(graph=self._graph)as session:
            return session.run(pca, feed_dict={self._X: self._data})


minist = input_data.read_data_sets('MNIST_data/')

tf_pca = TF_PCA(minist.train.images)
tf_pca.fit()
pca = tf_pca.reduce(keep_info=0.1)# 保存的信息占比
print('original data shape', minist.train.images.shape)
print('reduced data shape ', pca.shape)
Set = sns.color_palette('Set2', 10)
color_mapping = {key: value for (key, value) in enumerate(Set)}
colors = list(map(lambda x: color_mapping[x], minist.train.labels))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=colors)
plt.show()