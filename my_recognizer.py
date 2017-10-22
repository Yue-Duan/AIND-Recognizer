import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    # raise NotImplementedError
    X_lengths = test_set.get_all_Xlengths()
    # loop through all test data
    for X, lengths in X_lengths.values():
        # save likelihood of words
        log_L = {}
        # save max score of all words
        best_score = float("-inf")
        # save word with best score as the recognizer guess
        best_guess = None
        for word, model in models.items():
            try:
                # get model score
                log_L[word] = model.score(X, lengths)
                # update best score and best guess
                if log_L[word] > best_score:
                    best_score = log_L[word]
                    best_guess = word
            except:
                # if the model cannot process the word
                log_L[word] = float("-inf")

        probabilities.append(log_L)
        guesses.append(best_guess)

    return probabilities, guesses
