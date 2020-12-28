import numpy as np
import util
import matplotlib.pyplot as plt
from linearclass.logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted

    # ****************************** PART A ******************************
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    t_train = 1 - t_train
    clf1 = LogisticRegression(eps=1e-15)
    clf1.fit(x_train, t_train)

    # Plot test data set
    fig1, ax1 = plt.subplots()
    ax1.plot(x_test[t_test == 1, -2], x_test[t_test == 1, -1], 'b+', linewidth=2)
    ax1.plot(x_test[t_test == 0, -2], x_test[t_test == 0, -1], 'g_', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x_test[:, -2]), max(x_test[:, -2]), 0.01)
    x2 = -(clf1.theta[0] / clf1.theta[2] + clf1.theta[1] / clf1.theta[2] * x1)
    ax1.plot(x1, x2, c='red', linewidth=2)

    ax1.set_xlim(x_test[:, -2].min() - .1, x_test[:, -2].max() + .1)
    ax1.set_ylim(x_test[:, -1].min() - .1, x_test[:, -1].max() + .1)

    # Add labels
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')

    # Save predictions on test set to save_path
    np.savetxt(output_path_true, clf1.predict(x_test))

    # ****************************** PART B ******************************
    _, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    _, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    clf2 = LogisticRegression()
    clf2.fit(x_train, y_train)

    # Plot test data set
    fig2, ax2 = plt.subplots()
    ax2.plot(x_test[t_test == 1, -2], x_test[t_test == 1, -1], 'b+', linewidth=2)
    ax2.plot(x_test[t_test == 0, -2], x_test[t_test == 0, -1], 'g_', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x_test[:, -2]), max(x_test[:, -2]), 0.01)
    x2 = -(clf2.theta[0] / clf2.theta[2] + clf2.theta[1] / clf2.theta[2] * x1)
    ax2.plot(x1, x2, c='red', linewidth=2)

    ax2.set_xlim(x_test[:, -2].min() - .1, x_test[:, -2].max() + .1)
    ax2.set_ylim(x_test[:, -1].min() - .1, x_test[:, -1].max() + .1)

    # Add labels
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')

    # Save predictions on test set to save_path
    np.savetxt(output_path_naive, clf2.predict(x_test))

    # ****************************** PART F ******************************
    x_val, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    alpha = np.mean(clf2.predict(x_val[y_val == 1]))

    # Plot test data set
    fig3, ax3 = plt.subplots()
    ax3.plot(x_test[t_test == 1, -2], x_test[t_test == 1, -1], 'b+', linewidth=2)
    ax3.plot(x_test[t_test == 0, -2], x_test[t_test == 0, -1], 'g_', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x_test[:, -2]), max(x_test[:, -2]), 0.01)
    x2 = -(clf2.theta[0] / clf2.theta[2] + clf2.theta[1] / clf2.theta[2] * x1
           + np.log((2 - alpha) / alpha) / clf2.theta[2])
    ax3.plot(x1, x2, c='red', linewidth=2)

    ax3.set_xlim(x_test[:, -2].min() - .1, x_test[:, -2].max() + .1)
    ax3.set_ylim(x_test[:, -1].min() - .1, x_test[:, -1].max() + .1)

    # Add labels
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')

    # Save predictions on test set to save_path
    np.savetxt(output_path_adjusted, clf2.predict(x_test)/alpha)

    # *** END CODE HERE


if __name__ == '__main__':
    main(train_path='train.csv',
         valid_path='valid.csv',
         test_path='test.csv',
         save_path='posonly_X_pred.txt')
