# 11.Mixture-model
 EM Algorithm for Gaussian Mixture Models (GMM)
 📊 EM Algorithm for Gaussian Mixture Models (GMM)
This project implements the Expectation-Maximization (EM) algorithm for estimating the parameters of multivariate Gaussian Mixture Models (GMMs). It supports a flexible number of mixture components and iteratively optimizes the mean vectors, covariance matrices, and mixing weights. Designed for applications in clustering and density modeling, the project highlights the underlying mathematics of EM—showcasing probabilistic responsibility computation and parameter updates. Ideal for students and practitioners in pattern recognition and probabilistic learning, this work enables deep exploration of soft clustering and Gaussian-based data modeling using real-world datasets.

![image](https://github.com/user-attachments/assets/386c7123-b956-4582-b037-45978e6b4435)

🧮 Typed Content:
Implement a Matlab/Python function realising expectation maximisation (EM) for multivariate Gaussian mixtures. The function call should be:
and the parameters and the outputs should be as follows:
data contains the training samples,
J is the number of components in the mixture,
epsilon is an optional difference threshold for continuing iteration,
mu is a matrix containing a mean vector for each of the generating Gaussians,
Sigma is a three-dimensional array of covariance matrices, where the covariance matrices are along the third dimension, and
P is a vector of probabilities (mixing parameters).
The distribution to model is

  its mixing parameter.

Verify your implementation using the given data: CSV, MAT.
Experiment with different number of components in the mixture and visualise the results.

Hints:
To estimate the unknowns, the following update rules can be used:
θ(t) represents the parameters of a specific component at iteration t.

For the implementation, you can start with a template: Matlab, Python
```python
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

```
```matlab
Description :
mplementation of the EM Algorithm for Gaussian Mixture Models
This code implements the Expectation-Maximization (EM) algorithm for multivariate Gaussian mixtures in Python. The function ArmanGolbidi_em_gmm(data, clusters, tolerance) takes the following inputs:

data: The dataset to cluster.
clusters (J): The number of Gaussian components in the mixture.
tolerance (epsilon): The threshold for convergence.
The function outputs the estimated parameters of the Gaussian mixture model:

means (mu): The center of each Gaussian component.
covariances (Sigma): The covariance matrices for each component.
mixing probabilities (P): The probability of each component in the mixture.
The code also includes visualization functions to display the data and Gaussian ellipses representing each component in the final clustering. The plot below shows the clustered data points and the Gaussian components fitted by the algorithm:
"
and my code : 


# Import necessary libraries

##################################################################################### Part One
# Libraries and dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Load data
ArmanGolbidi_data = np.loadtxt('gmmdata.csv', delimiter=',')
ArmanGolbidi_data = ArmanGolbidi_data.T
print(f'Data shape: {ArmanGolbidi_data.shape}')

##################################################################################### Part Two
# EM algorithm function for Gaussian Mixture Models
def ArmanGolbidi_em_gmm(data, clusters, tolerance=1e-3):
    """
    EM algorithm for fitting a Gaussian Mixture Model to data.
    """
    # Data dimensions and initial parameters
    features, samples = data.shape
    means = np.random.rand(features, clusters) * np.max(data, axis=1).reshape(-1, 1)
    covariances = np.array([np.eye(features) for _ in range(clusters)]).transpose(1, 2, 0)
    mix_probabilities = np.full(clusters, 1 / clusters)

    # EM iteration
    while True:
        responsibilities = np.zeros((samples, clusters))
        for j in range(clusters):
            try:
                responsibilities[:, j] = mix_probabilities[j] * multivariate_normal.pdf(data.T, mean=means[:, j], cov=covariances[:, :, j])
            except np.linalg.LinAlgError:
                covariances[:, :, j] += np.eye(features) * 1e-6
                responsibilities[:, j] = mix_probabilities[j] * multivariate_normal.pdf(data.T, mean=means[:, j], cov=covariances[:, :, j])

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # Update parameters
        N_k = responsibilities.sum(axis=0)
        new_means = np.dot(data, responsibilities) / N_k

        new_covariances = np.zeros((features, features, clusters))
        for j in range(clusters):
            diff = data - new_means[:, j].reshape(-1, 1)
            new_covariances[:, :, j] = (responsibilities[:, j] * diff).dot(diff.T) / N_k[j]
            new_covariances[:, :, j] += np.eye(features) * 1e-6

        new_mix_probabilities = N_k / samples

        # Convergence check
        if np.linalg.norm(means - new_means) < tolerance:
            break

        means, covariances, mix_probabilities = new_means, new_covariances, new_mix_probabilities

    # Plot final results
    fig, ax = plt.subplots()
    ArmanGolbidi_visualise_data(clusters, means, covariances, data, ax)

    return means, covariances, mix_probabilities

##################################################################################### Part Three
# Define Visualization Functions
import matplotlib.patches as patches

def ArmanGolbidi_plot_gaussian_ellipsoid(mean, covariance, nstd=2, ax=None, **kwargs):
    """
    Plot an ellipse representing a Gaussian component's covariance.
    """
    if ax is None:
        ax = plt.gca()

    # Ellipsoid properties
    vals, vecs = np.linalg.eigh(covariance)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)

    ellip = patches.Ellipse(xy=(mean[0], mean[1]), width=width, height=height, angle=theta, edgecolor='purple', facecolor='none', **kwargs)
    ax.add_patch(ellip)

def ArmanGolbidi_visualise_data(clusters, means, covariances, data, ax):
    """
    Visualize the data and Gaussian components.
    """
    ax.clear()
    ax.scatter(data[0, :], data[1, :], s=5, label='Data Points', color='skyblue')
    ax.axis('equal')
    for i in range(clusters):
        ArmanGolbidi_plot_gaussian_ellipsoid(means[:, i], covariances[:, :, i], 2, ax)
    ax.legend()
    ax.set_title("Gaussian Mixture Model Clustering", color='blue')
    plt.show()

######################################################################################### Part Four
# We take transpose of T for shape-related issues
ArmanGolbidi_data = ArmanGolbidi_data.T
print(f'Transposed data shape: {ArmanGolbidi_data.shape}')

######################################################################################### Part Five
# Set number of clusters and tolerance
ArmanGolbidi_clusters = 3
ArmanGolbidi_tolerance = 1e-3

# Apply EM algorithm and display results
means, covariances, mix_probabilities = ArmanGolbidi_em_gmm(ArmanGolbidi_data, ArmanGolbidi_clusters, ArmanGolbidi_tolerance)
print("Means:", means)
print("Covariances:", covariances)
print("Mixing Probabilities:", mix_probabilities)
```
