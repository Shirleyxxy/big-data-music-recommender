#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pyarrow
import time

def main(item_factors_path, tags_path, tracks_path):

    np.random.seed(1004)

    item_factors = pd.read_parquet(item_factors_path)
    tags = pd.read_parquet(tags_path)
    tracks = pd.read_parquet(tracks_path)
    feat_num = len(item_factors.features[0])
    tracks.track_label = tracks.track_label.astype(int)
    # create a mapping dictionary
    track_dict = dict(zip(tracks.track_label, tracks.track_id))
    item_factors = item_factors.replace({'id': track_dict})
    tags = tags.drop(['score'], axis = 1)
    item_factors = pd.merge(item_factors, tags, left_on = ['id'], right_on = ['track_id'])
    item_factors = item_factors.drop(columns = ['id', 'track_id'], axis = 1)
    # get 15 most popular tags
    tags_list = item_factors['tag'].value_counts()[:15].index.tolist()
    print(tags_list)
    item_factors = item_factors.loc[item_factors['tag'].isin(tags_list)]
    feat_cols = ['feat ' + str(i) for i in range(feat_num)]
    item_factors[feat_cols] = pd.DataFrame(item_factors.features.tolist(), index = item_factors.index)
    item_factors = item_factors.drop(['features'], axis = 1)
    # calculate tSNE scores
    features = item_factors[feat_cols].values
    time_start = time.time()
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 20, n_iter = 300)
    tsne_results = tsne.fit_transform(features)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    item_factors['tsne-2d-one'] = tsne_results[:, 0]
    item_factors['tsne-2d-two'] = tsne_results[:, 1]
    # plotting
    plt.figure(figsize = (10, 10))

    sns.scatterplot(
        x = 'tsne-2d-one', y = 'tsne-2d-two',
        hue = 'tag',
        palette = sns.color_palette('RdBu', 15),
        data = item_factors,
        legend = 'full',
        alpha = 0.3
    )

    plt.title('t-SNE Visualization of Tracks by Tags')
    plt.show()


if __name__ == '__main__':

    item_factors_path = sys.argv[1]
    tags_path = sys.argv[2]
    tracks_path = sys.argv[3]

    main(item_factors_path, tags_path, tracks_path)

