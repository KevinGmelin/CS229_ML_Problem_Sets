import numpy as np
import util
import matplotlib.pyplot as plt


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path

    # Creates logistic regression classifer
    clf = LogisticRegression()

    # Train the model using training data
    clf.fit(x_train, y_train)

    # Load validation data set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)

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


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        # Keep updating theta until theta converges or the max number of iterations is reached
        for iteration in range(self.max_iter):
            loss_gradient = np.empty(self.theta.size)  # Gradient of average log loss function
            hessian = np.empty((self.theta.size, self.theta.size)) # Hessian matrix of average log loss function

            # Construct gradient vector of the average log loss function with respect to theta
            for j in range(loss_gradient.size):
                partial_diff = 0
                for i in range(y.size):
                    partial_diff += (y[i] - sigmoid(self.theta.dot(x[i]))) * x[i][j]
                partial_diff /= -y.size
                loss_gradient[j] = partial_diff

            # Construct hessian matrix of the average log loss function with respect to theta
            for j, k in np.ndindex(hessian.shape):
                partial_secondary_diff = 0
                for i in range(y.size):
                    h = sigmoid(self.theta.dot(x[i]))
                    partial_secondary_diff += h * (1-h) * x[i][j] * x[i][k]
                partial_secondary_diff /= y.size
                hessian[j][k] = partial_secondary_diff

            # Store the prior value of theta
            oldTheta = self.theta.copy()

            # Apply newtons method to update theta
            try:
                self.theta -= np.linalg.inv(hessian).dot(loss_gradient)
            except np.linalg.LinAlgError as error:
                if 'Singular matrix' in str(error):
                    print("Error: Cannot invert hessian matrix. Hessian is singular.")
                else:
                    print("Error inverting hessian matrix.")

            # Return if the new theta vector is within epsilon of the old theta vector
            # Signifies that theta has converged
            if np.linalg.norm(self.theta - oldTheta, 1) < self.eps:
                return

        print("Warning: Theta failed to converged within the given max number of iterations.")
        return
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.array([sigmoid(self.theta.dot(xi)) for xi in x])
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
