# This is just a simple function to execute the code in general
# This file does not need to be submitted. Fiddle with it any way you see fit.
from functools import partial
from sklearn.metrics import accuracy_score, precision_score, recall_score
from c1_cleveland_data_preprocessor import *
from c2_data_balancing import balance_dataset
from c3_naive_bayes import NaiveBayes
from c4_ethical_evaluation import Evaluator

# We load the dataset. Change the path as you see fit.
dataset_path = "processed_cleveland_dataset.csv"
dataset = read_data(dataset_path)

# That's the name of the class variable in this dataset
class_name = 'target'

# We preprocess the dataset
training_data, testing_data = preprocess(dataset, class_name)

# Component 2 - Balance the training data
balanced_training_data = balance_dataset(training_data, class_name)

# We extract some data for booting up our classifier
full_data = pd.concat([training_data, testing_data])
feature_info = {col: sorted(full_data[col].unique().tolist()) for col in full_data.columns}
class_values = feature_info.pop(class_name, None)

# Component 3 - Classifier is created, trained on balanced data, and predictions are made
nb_classifier = NaiveBayes((class_name, class_values), feature_info)
nb_classifier.fit(balanced_training_data)
classified_data = nb_classifier.predict(testing_data)

# Evaluations are made based on the predictions
true_classes = classified_data[class_name]
predicted_classes = classified_data["PredictedClass"]
evaluator = Evaluator(class_values)

print("\n=== Overall Classification Performance ===")
print(evaluator.evaluate_classification(true_classes, predicted_classes))

print("\n=== Group Fairness Evaluation (Sex) ===")
print(evaluator.compute_group_fairness_ethical_evaluation(
    true_classes,
    predicted_classes,
    testing_data['sex']
))

# Component 4 - Individual fairness evaluated via counterfactual fairness
print("\n=== Counterfactual Fairness Evaluation ===")
counterfactual_results = evaluator.compute_counterfactual_fairness(
    testing_data=testing_data,
    classifier=nb_classifier,
    class_name=class_name,
    protected_attribute='sex',
)
print(counterfactual_results)