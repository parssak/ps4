'''
HW4 Q3

Implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class. You may iterate over
    the possible digits (0 to 9), but otherwise make sure that your code
    is vectorized.

    Arguments
        train_data: size N x 64 numpy array with the images
        train_labels: size N numpy array with corresponding labels
    
    Returns
        means: size 10 x 64 numpy array with the ith row corresponding
               to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for digit in digits:
        subset = train_data[train_labels == digit]
        means[digit] = np.mean(subset, axis=0)

    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class. You may iterate over
    the possible digits (0 to 9), but otherwise make sure that your code
    is vectorized.

    Arguments
        train_data: size N x 64 numpy array with the images
        train_labels: size N numpy array with corresponding labels
    
    Returns
        covariances: size 10 x 64 x 64 numpy array with the ith row corresponding
               to the covariance matrix estimate for digit class i
    '''
    # Initialize array to store covariances
    covariances = np.zeros((10, 64, 64))

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for digit in digits:
        subset = train_data[train_labels == digit]
        centered_subset = subset - np.mean(subset, axis=0)
        covariances[digit] = np.dot(centered_subset.T, centered_subset)

    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood log p(x|t). You may iterate over
    the possible digits (0 to 9), but otherwise make sure that your code
    is vectorized.

    Arguments
        digits: size N x 64 numpy array with the images
        means: size 10 x 64 numpy array with the 10 class means
        covariances: size 10 x 64 x 64 numpy array with the 10 class covariances
    
    Returns
        likelihoods: size N x 10 numpy array with the ith row corresponding
               to logp(x^(i) | t) for t in {0, ..., 9}
    '''
    N = digits.shape[0]
    likelihoods = np.zeros((N, 10))
    # == YOUR CODE GOES HERE ==
    # ====
    return likelihoods


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood log p(t|x). Make sure that your code
    is vectorized.

    Arguments
        digits: size N x 64 numpy array with the images
        means: size 10 x 64 numpy array with the 10 class means
        covariances: size 10 x 64 x 64 numpy array with the 10 class covariances
    
    Returns
        likelihoods: size N x 10 numpy array with the ith row corresponding
               to logp(t | x^(i)) for t in {0, ..., 9}
    '''

    # == YOUR CODE GOES HERE ==
    # ====
    pass

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class. 
    Make sure that your code is vectorized.

    Arguments
        digits: size N x 64 numpy array with the images
        means: size 10 x 64 numpy array with the 10 class means
        covariances: size 10 x 64 x 64 numpy array with the 10 class covariances
    
    Returns
        pred: size N numpy array with the ith element corresponding
               to argmax_t log p(t | x^(i))
    '''
    # Compute and return the most likely class

    # == YOUR CODE GOES HERE ==
    # ====
    pass

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(t^(i) | x^(i)) )

    i.e. the average log likelihood that the model assigns to the correct class label.

    Arguments
        digits: size N x 64 numpy array with the images
        labels: size N x 10 numpy array with the labels
        means: size 10 x 64 numpy array with the 10 class means
        covariances: size 10 x 64 x 64 numpy array with the 10 class covariances
    
    Returns
        average conditional log-likelihood.
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    assert len(digits) == len(labels)
    sample_size = len(digits)
    total_prob = 0
    for i in range(sample_size):
        total_prob += cond_likelihood[i][int(labels[i])]

    return total_prob/sample_size



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data()

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    train_log_llh = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_log_llh = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    print('Train average conditional log-likelihood: ', train_log_llh)
    print('Test average conditional log-likelihood: ', test_log_llh)

    train_posterior_result = classify_data(train_data, means, covariances)
    test_posterior_result = classify_data(test_data, means, covariances)

    train_accuracy = np.mean(train_labels.astype(int) == train_posterior_result)
    test_accuracy = np.mean(test_labels.astype(int) == test_posterior_result)

    print('Train posterior accuracy: ', train_accuracy)
    print('Test posterior accuracy: ', test_accuracy)

    for i in range(10):
        (e_val, e_vec) = np.linalg.eig(covariances[i])
        # In particular, note the axis to access the eigenvector
        curr_leading_evec = e_vec[:,np.argmax(e_val)].reshape((8,8))
        plt.subplot(3,4,i+1)
        plt.imshow(curr_leading_evec, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
