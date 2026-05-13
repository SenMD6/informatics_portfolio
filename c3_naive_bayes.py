import copy

import numpy
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder


# You are allowed to change anything you see fit here, as long as some kind of naive Bayes
# (from any kind of library etc.) is used :)
class NaiveBayes:

    # This function simply initializes an instance of NaiveBayes class. The constructor takes at input:
    # - class_info - pair that contains the name of the class column and its permitted values
    # - feature_info - dictionary that states attribute names and their permitted values
    # Admittedly, that data is not used much anymore due to sklearn. Feel free to purge it if you wish.
    #
    # For the purpose of this portfolio, we are using CategoricalNB implementation from sklearn. For this reason,
    # we are also using an OrdinalEncoder, so that any string columns can be mapped into integers (it's a
    # peculiarity of the sklearn implementations of NaiveBayes).

    def __init__(self, class_info: tuple[str, list[str]], feature_info: dict[str, list[str]]):
        self.class_info = class_info
        self.feature_info = feature_info
        self.inner_nb = CategoricalNB(alpha=1e-10, force_alpha=True)
        self.encoder = OrdinalEncoder()
        # You can add further variables/attributes/etc. here

    # This function trains the model, aka calculates all the necessary probabilities that a naive Bayes model needs.
    #
    # The training means fitting both the inner CategoricalNB and the inner OrdinalEncoder that is used to adapt
    # the code to sklearn.
    def fit(self, training_data: pd.DataFrame):
        class_name = self.class_info[0]
        features_train = self.encoder.fit_transform(training_data.drop(class_name, axis=1, inplace=False))
        class_train = training_data[class_name]
        self.inner_nb.fit(features_train, class_train)

    # This function predicts the classes for entries in the training_data and produces an extended data frame.
    # At input, it takes:
    # - testing_data - a pandas DataFrame that contains all the attribute values and class value for a given entry
    # The function outputs:
    # classified_data - a pandas DataFrame which expands the training_data by adding the "PredictedClass" column
    #                   that for every entry states the class value predicted for that entry. In case of ties,
    #                   the chosen class is the one that appears earlier alphabetically.
    def predict(self, testing_data: pd.DataFrame) -> pd.DataFrame:
        class_name = self.class_info[0]
        # As a precaution, we drop the class column. Also, we need to use the previously trained encosder
        # to transform the testing data.
        features_test = self.encoder.transform(testing_data.drop(class_name, axis=1, inplace=False, errors='ignore'))
        predicted_class = self.inner_nb.predict(features_test)
        classified_data = copy.deepcopy(testing_data)
        classified_data['PredictedClass'] = predicted_class
        return classified_data

    # The function returns the probability of a given class value. You can assume
    # that this function simply retrieves the desired probability after training rather than
    # recomputes them from scratch. A value of 0 should be returned if no training took place.
    # At input, it takes:
    # - class_value - the class value for which we want to calculate the probability
    # The function outputs:
    # - probability - float representing the probability of the given class value
    def retrieve_class_probability(self, class_value: str) -> float:
        class_index = numpy.where(self.inner_nb.classes_ == class_value)[0][0]
        class_probability = numpy.exp(self.inner_nb.class_log_prior_[class_index])
        return class_probability

    # The function returns the conditional probably of a feature value assuming a given class value. You can assume
    # that this function simply retrieves the desired probability after training rather than
    # recomputes them from scratch. A value of 0 should be returned if no training took place.
    # At input, it takes:
    # - class_value - the class value on which the feature_value is conditional
    # - feature_name - the name of the feature we want to calculate for
    # - feature_value - the feature value we want to calculate the conditional probability for
    # The function outputs:
    # - probability - float representing the calculated conditional probability
    #
    def retrieve_conditional_probability(self, class_value: str, feature_name: str, feature_value: str) -> float:
        class_index = numpy.where(self.inner_nb.classes_ == class_value)[0][0]
        feature_index = self.encoder.feature_names_in_.tolist().index(feature_name)
        feature_value_index = self.encoder.categories_[feature_index].tolist().index(feature_value)

        log_p = self.inner_nb.feature_log_prob_[feature_index][class_index, feature_value_index]
        return numpy.exp(log_p)
