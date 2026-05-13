import copy
import numpy
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import OrdinalEncoder

# using ComplementNB instead of CategoricalNB because
# the dataset is imbalanced.
# some disease stages and demographic groups appear less often,
# so ComplementNB can help reduce bias toward majority groups.
# OrdinalEncoder changes categorical values into numbers
# because sklearn Naive Bayes models need numerical input.

class NaiveBayes:

    def __init__(self, class_info: tuple[str, list[str]], feature_info: dict[str, list[str]]):
        self.class_info = class_info
        self.feature_info = feature_info
        # complementNB classifier
        # alpha=1.0 adds Laplace smoothing
        self.inner_nb = ComplementNB(alpha=1.0, force_alpha=True)
        # encodes strings into numbers
        self.encoder = OrdinalEncoder()

    # trains the model
    def fit(self, training_data: pd.DataFrame):
        class_name = self.class_info[0]
        # remove class column
        features_train = self.encoder.fit_transform(
            training_data.drop(class_name, axis=1, inplace=False)
        )
        class_train = training_data[class_name]
        # train classifier
        self.inner_nb.fit(features_train, class_train)

    # predicts classes for testing data
    def predict(self, testing_data: pd.DataFrame) -> pd.DataFrame:
        class_name = self.class_info[0]
        # encode testing data
        features_test = self.encoder.transform(
            testing_data.drop(class_name, axis=1, inplace=False, errors='ignore')
        )
        # predict classes
        predicted_class = self.inner_nb.predict(features_test)
        # copy original dataset
        classified_data = copy.deepcopy(testing_data)
        # add predictions
        classified_data['PredictedClass'] = predicted_class
        return classified_data

    # returns probability of a class
    def retrieve_class_probability(self, class_value: str) -> float:
        class_index = numpy.where(
            self.inner_nb.classes_ == class_value
        )[0][0]
        class_probability = numpy.exp(
            self.inner_nb.class_log_prior_[class_index]
        )
        return class_probability

    # returns conditional probability
    # kept mainly for compatibility with the template
    def retrieve_conditional_probability(
        self,
        class_value: str,
        feature_name: str,
        feature_value: str
    ) -> float:
        try:
            class_index = numpy.where(
                self.inner_nb.classes_ == class_value
            )[0][0]
            feature_index = self.encoder.feature_names_in_.tolist().index(
                feature_name
            )
            feature_value_index = self.encoder.categories_[
                feature_index
            ].tolist().index(feature_value)
            log_p = self.inner_nb.feature_log_prob_[
                feature_index
            ][class_index, feature_value_index]
            return numpy.exp(log_p)
        except:
            return 0.0