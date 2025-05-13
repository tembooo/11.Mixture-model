import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def em_gmm(data, J):
    #EM_GMM Gaussian mixture modelling by using expectation maximisation
    
    # Initialisation
    l, N = data.shape

    # Initialize means, covariances, and probabilities
    mu = np.zeros((l, J))       # mixture component means
    Sigma = np.zeros((l, l, J)) # mixture component covariances
    P = np.full((J, 1), 1/J)    # mixture component probabilities (mixing parameter)

    # Initialize the mixture components with mean and covariance
    for i in range(J):
        ...

    
    iterInd = 1

    fig, ax = plt.subplots()
    visualise_data(J, mu, Sigma, data, ax)

    # Estimate the parameters
    while True: # iterate forever (till the end criterion is met)
        # Pre-allocate the next values
        newmu = np.zeros((l, J))
        newP = np.zeros((J, 1))
        newSigma = np.zeros((l, l, J))

        for j in range(J): # for all mixture components
            # Update the mean
            # ...
            
            newmu[:, j] = ...

            # Update the covariance
            # ...
            

            newSigma[:, :, j] = ...

            # Update the probability in the mixture
            newP[j] = ...

            

        # Are the means moving: is it time to end the iteration?
        if np.linalg.norm(mu - newmu) < 0.01:
            break

        # Update the variables for the next iteration
        mu = newmu
        Sigma = newSigma
        P = newP
        iterInd += 1

        visualise_data(J, mu, Sigma, data, ax)
        plt.draw()
        plt.pause(0.0)

    visualise_data(J, mu, Sigma, data, ax)

# Helper function for determining P(j|\mathbf{x}_k; \Theta(t)
def pjxk(j, xk, mu, Sigma, P, J):
    ...
    return ...

# Helper function for determining p(\mathbf{x}_k; \Theta(t))
def pxk(xk, muj, Sigmaj):
    ...
    return ...

def visualise_data(J, mu, Sigma, data, ax):
    ax.clear()
    ax.scatter(data[0, :], data[1, :], s=1)
    ax.axis('equal')
    for i in range(J):
        plot_gaussian_ellipsoid(mu[:, i], Sigma[:, :, i], 2, ax, fill = False)
    # Display the updated figure
    plt.show(block=False)

def plot_gaussian_ellipsoid(mu, Sigma, nstd=2, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    vals, vecs = np.linalg.eigh(Sigma)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    
    ellip = plt.matplotlib.patches.Ellipse(mu, width, height, theta, **kwargs)

    ax.add_patch(ellip)
