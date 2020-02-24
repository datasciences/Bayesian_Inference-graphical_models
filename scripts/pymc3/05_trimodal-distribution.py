'''
Create a trimodal distribution with different number of
data points.
Input: n_cluster:number of datapoints
       mean
       std: standard deviation

'''

import numpy as np
import seaborn as sns

def main():
    # Create a mixture distribution with 3 means and std
    # There has to be 90, 50 and 75 draws from each of these clusters
    clusters = 3

    # number of points in each cluster
    n_cluster = [90, 50, 75]

    # total number of points in the mixture distribution
    n_total = sum(n_cluster)

    # mean and std of the 3 clusters
    means = [9, 21, 35]
    std_devs = [2, 2, 2]

    # Create a numpy array with the mean as many times you want to sample
    x = np.repeat(means, n_cluster)
    y = np.repeat(std_devs, n_cluster)

    # create a numpy array of the draws
    mix = np.random.normal(x, y)

    # plot the distribution
    sns.kdeplot(mix, shade = True)
    plt.xlabel('mixture distribution', fontsize = 14)
    plt.ylabel('density', fontsize = 14)


if __name__ == '__main__':
    main()
