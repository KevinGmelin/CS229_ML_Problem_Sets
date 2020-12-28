import numpy as np
import util
import matplotlib.pyplot as plt


def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path

    # Create and train a poisson regression model
    clf = PoissonRegression(step_size=1e-5, eps=1e-5)
    clf.fit(x_train, y_train)

    # Create array of predictions from the validation set input data
    y_pred = clf.predict(x_val)

    # Plot true counts vs predicted counts from the validation set
    fig, ax = plt.subplots()
    ax.plot(y_val, y_pred, 'bx', linewidth=2)

    # Plot the ideal distribution (true counts = expected counts)
    x1 = np.arange(0, max(y_val), 0.01)
    x2 = x1
    ax.plot(x1, x2, c='red', linewidth=2, label='Ideal Distribution')

    # Set axis limits
    ax.set_xlim(0, y_val.max() + .1)
    ax.set_ylim(0, y_pred.max() + .1)

    # Add labels and a legend
    ax.set_xlabel('True Count')
    ax.set_ylabel('Predicted Count')
    ax.legend()

    # Save predictions on validation set to a text file
    np.savetxt(save_path, y_pred)

    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        # Keep updating theta until theta converges or the max number of iterations is reached
        for iteration in range(self.max_iter):

            # Store the prior value of theta
            oldTheta = self.theta.copy()

            # Perform batch gradient descent to update theta
            for x_i, y_i in zip(x, y):
                self.theta += self.step_size * (y_i - np.exp(self.theta.dot(x_i))) * x_i

            # Return if the new theta vector is within epsilon of the old theta vector
            # Signifies that theta has converged
            if np.linalg.norm(self.theta - oldTheta, 1) < self.eps:
                return

        print("Warning: Theta failed to converged within the given max number of iterations.")
        return
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
