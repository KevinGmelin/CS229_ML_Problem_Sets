import util
import matplotlib.pyplot as plt
from gda import GDA
from logreg import LogisticRegression
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def main(train_path, valid_path):
    """
        Compares the performance between GDA and logistic regression classifiers
    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
    """

    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # x_train[:, 1] = x_train[:, 1] / np.exp(x_train[:, 0])

    # Creates gaussian discriminant analysis classifer
    gda_clf = GDA()
    log_clf = LogisticRegression()

    # Train the models using training data
    gda_clf.fit(x_train, y_train)
    log_clf.fit(util.add_intercept(x_train), y_train)

    # Load validation data set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=False)
    # x_val[:, 1] = x_val[:, 1] / np.exp(x_val[:, 0])

    # Plot validation data set
    fig, ax = plt.subplots()
    ax.plot(x_val[y_val == 1, -2], x_val[y_val == 1, -1], 'bx', linewidth=2)
    ax.plot(x_val[y_val == 0, -2], x_val[y_val == 0, -1], 'go', linewidth=2)

    # Plot decision boundaries (found by solving for theta^T x = 0)
    x1 = np.arange(min(x_val[:, -2]), max(x_val[:, -2]), 0.01)
    x2_gda = -(gda_clf.theta[0] / gda_clf.theta[2] + gda_clf.theta[1] / gda_clf.theta[2] * x1)
    x2_log = -(log_clf.theta[0] / log_clf.theta[2] + log_clf.theta[1] / log_clf.theta[2] * x1)
    ax.plot(x1, x2_gda, c='red', linewidth=2, label="GDA Decision Boundary")
    ax.plot(x1, x2_log, c='black', linewidth=2, label="Logistic Regression Decision Boundary")

    # Plot GDA Confidence Ellipses
    confidence_ellipse(gda_clf.mu_0, gda_clf.sigma, ax, edgecolor='green', linewidth=2)
    confidence_ellipse(gda_clf.mu_1, gda_clf.sigma, ax, edgecolor='blue', linewidth=2)

    ax.set_xlim(x_val[:, -2].min()-.1, x_val[:, -2].max()+.1)
    ax.set_ylim(x_val[:, -1].min()-.1, x_val[:, -1].max()+.1)

    # Add labels and a legend
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()


def confidence_ellipse(mu, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse.
    Based on the function from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html.

    Args:
        mu: array-like, shape (2, )
            Mean of the gaussian data

        cov: array-like, shape (2, 2)
            Covariance matrix

        ax: matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std: float
            The number of standard deviations to determine the ellipse's radii.

        **kwargs:
            Forwarded to `~matplotlib.patches.Ellipse

    Returns:
        matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv')