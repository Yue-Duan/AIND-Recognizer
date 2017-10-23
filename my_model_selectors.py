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

        best_component = self.n_components[np.argmin(BIC_scores)] if BIC_scores else self.n_constant
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
        results = []
        antiRes = []
        try:
            for n_component in self.n_components:
                # fit model on train
                model = self.base_model(n_component)
                # score on train
                logL = model.score(self.X, self.lengths)
                # score on everything else
                # modified as suggested
                antiLogL = np.mean( [ model.score(*self.hwords[word]) for word in self.words if word != self.this_word ] )
                # antiLogL = 0.0
                # word_count = 0
                # for word in self.hwords:
                #     if word == self.this_word:
                #         continue
                #     X, lengths = self.hwords[word]
                #     antiLogL += model.score(X, lengths)
                #     word_count += 1
                # # normalize
                # antiLogL = antiLogL/float(word_count)

                # calculate DIC and save result
                DIC = logL - antiLogL
                DIC_scores.append(DIC)
                results.append(logL)
                antiRes.append(antiLogL)

        except Exception as e:
            pass
            # print('!')

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


        for n_component in self.n_components:
            # print(n_component)
            if len(self.sequences) < 2:
                # for words with not long enough sequences, train and score on all data
                # print('only 1 sample')
                try:
                    model = self.base_model(n_component)
                    model_score = model.score(self.X, self.lengths)
                except Exception as e:
                    model_score = float("-inf")
                fold_mean_scores.append(model_score)
            else:
                fold_score_list = []
                # split to 3 folds if possible
                # k-fold split
                split_method = KFold(n_splits=min(len(self.sequences), 3))
                for train_idx, test_idx in split_method.split(self.sequences):
                    train_X, train_lengths = combine_sequences(train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(test_idx, self.sequences)
                    try:
                        # train model using training data only
                        cv_model = GaussianHMM(n_component, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                        # calculate test score for this fold
                        fold_score_list.append(cv_model.score(test_X, test_lengths))
                    except Exception as e:
                        pass
                # Compute mean of all fold scores
                fold_mean_scores.append(np.mean(fold_score_list))

        best_component = self.n_components[np.argmax(fold_mean_scores)] if fold_mean_scores else self.n_constant
        return self.base_model(best_component) #, fold_mean_scores # for sanity check
