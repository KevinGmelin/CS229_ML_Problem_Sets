import collections
import numpy as np
import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words_unnormalized = message.split(' ')
    words = [str.lower(word) for word in words_unnormalized]
    return words
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    # Create dictionary where each key is a word found in the list and each value is the number of occurrences
    # of that word in the messages
    counter = collections.Counter()
    for msg in messages:
        for word in get_words(msg):
            counter[word] += 1

    # Remove words with less than 5 occurrences
    # For the remaining words, change their value to be an index starting from 0
    index = 0
    dictionary = {}
    for word, cnt in counter.items():
        if cnt >= 5:
            dictionary[word] = index
            index = index + 1
    return dictionary
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    feature_array = np.zeros((len(messages), len(word_dictionary)))
    for msg_idx, msg in enumerate(messages):
        for word in get_words(msg):
            try:
                feature_array[msg_idx, word_dictionary[word]] += 1
            except KeyError:
                continue
    return feature_array
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    p_spam_prior = np.mean(labels == 1)

    # Probability of a word being in a message given that the message is spam
    p_word_given_spam = matrix[labels == 1].sum(0)
    p_word_given_spam += 1  # Add 1 for laplace smoothing
    p_word_given_spam /= (p_word_given_spam.sum() + len(p_word_given_spam))

    # Probability of a word being in a message given that the message is not spam
    p_word_given_ham = matrix[labels == 0].sum(0)
    p_word_given_ham += 1  # Add 1 for laplace smoothing
    p_word_given_ham /= (p_word_given_ham.sum() + len(p_word_given_ham))

    return p_spam_prior, p_word_given_spam, p_word_given_ham
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containing the predictions from the model
    """
    # *** START CODE HERE ***
    p_spam_prior, p_word_given_spam, p_word_given_ham = model

    p_ham_prior = 1 - p_spam_prior
    num_messages = matrix.shape[0]
    dict_size = matrix.shape[1]

    # Loops over and classifies all messages as either spam or ham (not spam)
    # 1 signifies spam, 0 signifies ham
    predictions = np.empty(num_messages)
    for i in range(num_messages):
        log_p_x_given_spam = 0
        log_p_x_given_ham = 0
        for j in range(dict_size):
            if matrix[i, j] != 0:
                log_p_x_given_spam += matrix[i, j] * np.log(p_word_given_spam[j])
                log_p_x_given_ham += matrix[i, j] * np.log(p_word_given_ham[j])

        # Use Bayes Rule to determine probability of spam given the message contents x
        p_spam_given_x = 1.0 / (1.0 + np.exp(log_p_x_given_ham - log_p_x_given_spam) * p_ham_prior / p_spam_prior)

        # Predict message will be spam if the posterior probability is 50 percent or more
        if p_spam_given_x >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0

    return predictions

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    p_spam_prior, p_word_given_spam, p_word_given_ham = model

    # Apply metric to each word's probabilities
    indicative_metric = np.log(p_word_given_spam / p_word_given_ham)

    # Find the indices of the top 5 words in descending order
    indices = np.argpartition(indicative_metric, -5)[-5:]  # Gets indices of 5 largest metrics
    indices[::-1].sort()  # Sort in descending order

    # Finds the 5 words that correspond with the 5 indices
    indicative_words = []
    for idx in indices:
        indicative_words.append(list(dictionary.keys())[list(dictionary.values()).index(idx)])

    return indicative_words

    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    accuracies_on_validation = np.empty(len(radius_to_consider))
    for idx, radius in enumerate(radius_to_consider):
        predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracies_on_validation[idx] = np.mean(predictions == val_labels)
    return radius_to_consider[accuracies_on_validation.argmax()]
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
