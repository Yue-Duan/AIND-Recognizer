import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_components = range(self.min_n_components, self.max_n_components + 1)

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # raise NotImplementedError
        BIC_scores = []
        try:
            for n in self.n_components:
                # print(n)
                model = self.base_model(n)
                log_L = model.score(self.X, self.lengths)
                p = n ** 2 + 2 * n * model.n_features - 1
                BIC_scores.append( -2 * log_L + p * math.log(n))
        except Exception as e:
            pass

        best_component = self.n_components[np.argmax(BIC_scores)] if BIC_scores else self.n_constant
        return self.base_model(best_component)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # raise NotImplementedError
        DIC_scores = []
        log_Xs = []
        try:
            M = len(self.n_components)
            for n_component in self.n_components:
                model = self.base_model(n_component)
                log_Xs.append(model.score(self.X, self.lengths))

            log_sum = sum(log_Xs)
            for log_X in log_Xs:
                DIC = log_X - (log_sum - log_X)*1./(M-1)
                DIC_scores.append(DIC)
        except Exception as e:
            pass

        best_component = self.n_components[np.argmax(DIC_scores)] if DIC_scores else self.n_constant
        return self.base_model(best_component)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # raise NotImplementedError
        # list to save different fold scores
        fold_mean_scores = []
        # k-fold split
        split_method = KFold()
        try:
            for n_component in self.n_components:
                model = self.base_model(n_component)
                # calculate model mean scores
                fold_score_list = []
                for _, test_idx in split_method.split(self.sequences):
                    # get test sequences
                    test_X, test_length = combine_sequences(test_idx, self.sequences)
                    # save model score to list
                    fold_score_list.append(model.score(test_X, test_length))
                # Compute mean of all fold scores
                fold_mean_scores.append(np.mean(fold_score_list))
        except Exception as e:
            pass

        best_component = self.n_components[np.argmax(fold_mean_scores)] if fold_mean_scores else self.n_constant
        return self.base_model(best_component)
