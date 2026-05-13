import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from fairlearn.metrics import MetricFrame


class Evaluator:

    # Simple init function. class_values represent all possible values of the class for which evaluator is going
    # to be used. Class_values are only used for labeling purposes. Feel free to modify this as you see fit.

    def __init__(self, class_values: list[str]):
        self.class_values = class_values

    # These are the macro and weighted metric counterparts from Portfolio 1. Instead of using the confusion matrix
    # as input, we are now using two separate series for actual and predicted classes so that this
    # can be done using libraries rather than the previous code.

    # You can modify the file any way you see fit, including using different libraries, variables, variable
    # types etc., as long as your final code as a whole makes sense, and the desired metrics are computed.

    # At input, functions take:
    #  - actual_classes, predicted_classes - series of class values representing actual and predicted
    #                                       classes of some dataset.
    #
    # As output, they produce float value representing the chosen metric.
    def compute_macro_precision(self, actual_classes: pd.Series, predicted_classes: pd.Series) -> float:
        return precision_score(actual_classes, predicted_classes, labels=self.class_values, average='macro',
                               zero_division=0.0)

    def compute_macro_recall(self, actual_classes: pd.Series, predicted_classes: pd.Series) -> float:
        return recall_score(actual_classes, predicted_classes, labels=self.class_values, average='macro',
                            zero_division=0.0)

    def compute_macro_f_measure(self, actual_classes: pd.Series, predicted_classes: pd.Series) -> float:
        return f1_score(actual_classes, predicted_classes, labels=self.class_values, average='macro', zero_division=0.0)

    def compute_weighted_precision(self, actual_classes: pd.Series, predicted_classes: pd.Series) -> float:
        return precision_score(actual_classes, predicted_classes, labels=self.class_values, average='weighted',
                               zero_division=0.0)

    def compute_weighted_recall(self, actual_classes: pd.Series, predicted_classes: pd.Series) -> float:
        return recall_score(actual_classes, predicted_classes, labels=self.class_values, average='weighted',
                            zero_division=0.0)

    def compute_weighted_f_measure(self, actual_classes: pd.Series, predicted_classes: pd.Series) ->float:
        return f1_score(actual_classes, predicted_classes, labels=self.class_values, average='weighted',
                        zero_division=0.0)

    def compute_standard_accuracy(self, actual_classes: pd.Series, predicted_classes: pd.Series) -> float:
        return accuracy_score(actual_classes, predicted_classes)

    def compute_balanced_accuracy(self, actual_classes: pd.Series, predicted_classes: pd.Series) -> float:
        return balanced_accuracy_score(actual_classes, predicted_classes)

    # This is the evaluate_classification counterpart from Portfolio 1. Class values are no longer given at input,
    # and the confusion matrix function is no longer needed. The function produces the output as a Series
    # rather than dictionary now. If you prefer to revert to dictionary version, simply replace
    # mf.overall with mf.overall.to_dict()

    # At input, function takes:
    #  - actual_classes, predicted_classes - series of class values representing actual and predicted
    #                                       classes of some dataset.
    #
    # As output, it produces a Series of metric values. The output series includes appropriate names as well.
    def evaluate_classification(self, actual_classes: pd.Series, predicted_classes: pd.Series) -> pd.Series:
        dummy_feature = ["All Samples"] * len(actual_classes)

        metrics = {'macro_precision': self.compute_macro_precision,
                   'macro_recall': self.compute_macro_recall,
                   'macro_f_measure': self.compute_macro_f_measure,
                   'weighted_precision': self.compute_weighted_precision,
                   'weighted_recall': self.compute_weighted_recall,
                   'weighted_f_measure': self.compute_weighted_f_measure,
                   'standard_accuracy': self.compute_standard_accuracy,
                   'balanced_accuracy': self.compute_balanced_accuracy
                   }
        mf = MetricFrame(
            metrics=metrics,
            y_true=actual_classes,
            y_pred=predicted_classes,
            sensitive_features=dummy_feature
        )

        return mf.overall

    # We are now doing ethical evaluation using Fairlearn. The standard metrics are now calculated per each
    # unique value of the selected feature, not for the dataset as a whole.
    # This leads to a non-aggregated group fairness evaluation style,
    # i.e., we get outcomes for each feature value and the comparison and judgement is up to us, rather than
    # various ratios already being calculated and aggregated for us.

    # You are free to expand on this to produce aggregated evaluation. However, the element needed for the portfolio
    # is about implementing an individual fairness approach.

    # At input, function takes:
    #  - actual_classes, predicted_classes - series of class values representing actual and predicted
    #                                       classes of some dataset
    #  - selected_feature - a series (of the same length as actual/predicted classes) of values of a given feature
    #                       according to which group fairness evaluation is to be done.
    #
    # As output, it produces a DataFrame of metric values. Columns correspond to various metrics, rows correspond
    # to unique values of the selected feature.
    def compute_group_fairness_ethical_evaluation(self, actual_classes: pd.Series, predicted_classes: pd.Series,
                                                  selected_feature: pd.Series) -> pd.DataFrame:
        metrics = {'macro_precision': self.compute_macro_precision,
                   'macro_recall': self.compute_macro_recall,
                   'macro_f_measure': self.compute_macro_f_measure,
                   'weighted_precision': self.compute_weighted_precision,
                   'weighted_recall': self.compute_weighted_recall,
                   'weighted_f_measure': self.compute_weighted_f_measure,
                   'standard_accuracy': self.compute_standard_accuracy,
                   'balanced_accuracy': self.compute_balanced_accuracy
                   }
        mf = MetricFrame(
            metrics=metrics,
            y_true=actual_classes,
            y_pred=predicted_classes,
            sensitive_features=selected_feature
        )

        return mf.by_group

    def compute_counterfactual_fairness(self, testing_data, classifier, class_name: str, 
                                        protected_attribute: str) -> dict:
        #copy of patient data to flip protected attribute on
        counterfactual_data = testing_data.copy()
        #flip protected attribute aka sex
        counterfactual_data[protected_attribute] = counterfactual_data[protected_attribute].map(
            {'male' : 'female' , 'female' : 'male'}
        )
        #run both versions thru classifier for predictions 
        original_predictions = classifier.predict(testing_data)['PredictedClass']
        counterfactual_predictions = classifier.predict(counterfactual_data)['PredictedClass']
        #compare the predictions for the original and flipped data
        changed_predictions = (original_predictions != counterfactual_predictions).sum()
        total_predictions = len(original_predictions)
        #calc as percentage
        changed_percentage = changed_predictions / total_predictions * 100

        return {
                'total_patients' : total_predictions,
                'changed_predictions' : int(changed_predictions),
                'unchanged_predictions' : int(total_predictions - changed_predictions),
                'changed_percentage' : round(changed_percentage, 2)
            }
        