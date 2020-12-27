import numpy as np
import util
import matplotlib.pyplot as plt


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path

    # Creates gaussian discriminant analysis classifer
    clf = GDA()

    # Train the model using training data
    clf.fit(x_train, y_train)

    # Load validation data set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=False)

    # Plot validation data set
    plt.figure()
    plt.plot(x_val[y_val == 1, -2], x_val[y_val == 1, -1], 'bx', linewidth=2)
    plt.plot(x_val[y_val == 0, -2], x_val[y_val == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x_val[:, -2]), max(x_val[:, -2]), 0.01)
    x2 = -(clf.theta[0] / clf.theta[2] + clf.theta[1] / clf.theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)

    plt.xlim(x_val[:, -2].min()-.1, x_val[:, -2].max()+.1)
    plt.ylim(x_val[:, -1].min()-.1, x_val[:, -1].max()+.1)

    # Add labels
    plt.xlabel('x1')
    plt.ylabel('x2')

    # Save predictions on eval set to save_path
    np.savetxt(save_path, clf.predict(x_val))

    # *** END CODE HERE ***


def sigmoid(x):
    """Calculates sigmoid function, which is commonly used in logistic regression.

    Args:
        x: Independent variable that is inputted to sigmoid function

    Returns:
        1/(1+exp(-x))
    """
    return 1/(1+np.exp(-x))


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.phi = None
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        n = y.size
        self.phi = np.count_nonzero(y) / n
        self.mu_0 = x[y == 0].sum(0)/np.count_nonzero(y == 0)
        self.mu_1 = x[y == 1].sum(0)/np.count_nonzero(y)
        mu_y = np.reshape(y == 0, (n, 1)) * self.mu_0 + np.reshape(y == 1, (n, 1)) * self.mu_1
        self.sigma = np.dot((x-mu_y).T, (x-mu_y))/n
        sigma_inverse = np.empty_like(self.sigma)

        try:
            sigma_inverse = np.linalg.inv(self.sigma)
        except np.linalg.LinAlgError as error:
            if 'Singular matrix' in str(error):
                print("Error: Cannot invert covariance matrix. Covariance is singular.")
            else:
                print("Error inverting covariance matrix.")

        theta_n = sigma_inverse.dot(self.mu_1 - self.mu_0)
        theta_0 = np.dot(self.mu_0, np.dot(sigma_inverse, self.mu_0))/2 - \
                  np.dot(self.mu_1, np.dot(sigma_inverse, self.mu_1))/2 + \
                  np.log(self.phi/(1-self.phi))
        self.theta = np.concatenate((np.array([theta_0]), theta_n))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        theta_0 = self.theta[0]
        theta_n = self.theta[1:]
        return np.array([sigmoid(theta_n.dot(xi) + theta_0) for xi in x])
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
