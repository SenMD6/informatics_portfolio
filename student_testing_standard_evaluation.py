import copy
import unittest
import pandas as pd

from c3_naive_bayes import *
from c4_ethical_evaluation import *

actual_class = pd.Series(
    ['Female', 'Male', 'Rodent', 'Female', 'Female', 'Rodent', 'Male', 'Male', 'Male', 'Female', 'Male', 'Rodent',
     'Rodent', 'Male', 'Primate', 'Male', 'Food', 'Food', 'Male', 'Food', 'Primate'],
    name='Class', index=list(range(0, 21)))
predicted_class = pd.Series(
    ['Female', 'Rodent', 'Rodent', 'Male', 'Rodent', 'Rodent', 'Female', 'Female', 'Rodent', 'Female', 'Male', 'Rodent',
     'Female', 'Male', 'Primate', 'Primate', 'Food', 'Primate', 'Food', 'Female', 'Rodent'],
    name='PredictedClass', index=list(range(0, 21)))
class_values = ['Female', 'Male', 'Primate', 'Rodent', 'Food']


class Task_2_Testing(unittest.TestCase):


    # Here we check if macro precision is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    def test8_compute_macro_precision(self):
        evaluator = Evaluator(class_values)
        student_result = evaluator.compute_macro_precision(copy.deepcopy(actual_class),copy.deepcopy(predicted_class))
        expected = 0.4523809523809524

        result = round_equal(student_result, expected)
        result_message = "Computing macro precision failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if macro recall is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def test9_compute_macro_recall(self):
        evaluator = Evaluator(class_values)
        student_result = evaluator.compute_macro_recall(copy.deepcopy(actual_class),copy.deepcopy(predicted_class))
        expected = 0.4666666666666667

        result = round_equal(student_result, expected)
        result_message = "Computing macro recall failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if macro f-measure is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def testx10_compute_macro_f_measure(self):
        evaluator = Evaluator(class_values)
        student_result = evaluator.compute_macro_f_measure(copy.deepcopy(actual_class),copy.deepcopy(predicted_class))
        expected = 0.42181818181818176

        result = round_equal(student_result, expected)
        result_message = "Computing macro f-measure failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

        # Here we check if weighted precision is calculated well
        # This function contains just one test.
        # The function simply checks one possible behaviour, and there are many more possible. More than that
        # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
        # or checked.

    def testx11_compute_weighted_precision(self):
        evaluator = Evaluator(class_values)
        student_result = evaluator.compute_weighted_precision(copy.deepcopy(actual_class),copy.deepcopy(predicted_class))
        expected = 0.502267573696145

        result = round_equal(student_result, expected)
        result_message = "Computing weighted precision failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

        # Here we check if weighted recall is calculated well
        # This function contains just one test.
        # The function simply checks one possible behaviour, and there are many more possible. More than that
        # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
        # or checked.

    def testx12_compute_weighted_recall(self):
        evaluator = Evaluator(class_values)
        student_result = evaluator.compute_weighted_recall(copy.deepcopy(actual_class),copy.deepcopy(predicted_class))
        expected = 0.42857142857142855

        result = round_equal(student_result, expected)
        result_message = "Computing weighted recall failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

        # Here we check if weighted f-measure is calculated well
        # This function contains just one test.
        # The function simply checks one possible behaviour, and there are many more possible. More than that
        # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
        # or checked.

    def testx13_compute_weighted_f_measure(self):
        evaluator = Evaluator(class_values)
        student_result = evaluator.compute_weighted_f_measure(copy.deepcopy(actual_class),copy.deepcopy(predicted_class))
        expected = 0.41385281385281386

        result = round_equal(student_result, expected)
        result_message = "Computing weighted f-measure failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if standard accuracy is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def testx14_compute_standard_accuracy(self):
        evaluator = Evaluator(class_values)
        student_result = evaluator.compute_standard_accuracy(copy.deepcopy(actual_class),copy.deepcopy(predicted_class))
        expected = 0.42857142857142855

        result = round_equal(student_result, expected)
        result_message = "Computing standard accuracy failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if balanced accuracy is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def testx15_compute_balanced_accuracy(self):
        evaluator = Evaluator(class_values)
        student_result = evaluator.compute_balanced_accuracy(copy.deepcopy(actual_class),copy.deepcopy(predicted_class))
        # Using a helper function for equality checking, given some funky data formats sometimes
        expected = 0.4666666666666666

        result = round_equal(student_result, expected)
        result_message = "Computing balanced accuracy failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

def frame_round_equal(data1: pd.DataFrame, data2: pd.DataFrame):
    nums1 = data1.to_numpy().flatten()
    nums2 = data2.to_numpy().flatten()
    return list_round_equal(nums1, nums2)


def list_round_equal(nums1, nums2):
    if len(nums1) != len(nums2):
        return False
    for i in range(0, len(nums1)):
        v1 = round(float(nums1[i]), 5)
        v2 = round(float(nums2[i]), 5)
        if v1 != v2:
            return False
    return True


def round_equal(nums1: float, nums2: float):
    v1 = round(float(nums1), 5)
    v2 = round(float(nums2), 5)
    if v1 != v2:
        return False
    return True


if __name__ == '__main__':
    unittest.main()
