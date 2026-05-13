import unittest
import pandas as pd

from c3_naive_bayes import NaiveBayes

mammal_dataset = pd.DataFrame([['human', 'yes', 'no', 'no', 'yes', 'mammal'],
                               ['python', 'no', 'no', 'no', 'no', 'non-mammal'],
                               ['salmon', 'no', 'no', 'yes', 'no', 'non-mammal'],
                               ['whale', 'yes', 'no', 'yes', 'no', 'mammal'],
                               ['frog', 'no', 'no', 'sometimes', 'yes', 'non-mammal'],
                               ['komodo_dragon', 'no', 'no', 'no', 'yes', 'non-mammal'],
                               ['bat', 'yes', 'yes', 'no', 'yes', 'mammal'],
                               ['pigeon', 'no', 'yes', 'no', 'yes', 'non-mammal'],
                               ['cat', 'yes', 'no', 'no', 'yes', 'mammal'],
                               ['leopard_shark', 'yes', 'no', 'yes', 'no', 'non-mammal'],
                               ['turtle', 'no', 'no', 'sometimes', 'yes', 'non-mammal'],
                               ['penguin', 'no', 'no', 'sometimes', 'yes', 'non-mammal'],
                               ['porcupine', 'yes', 'no', 'no', 'yes', 'mammal'],
                               ['eel', 'no', 'no', 'yes', 'no', 'non-mammal'],
                               ['salamander', 'no', 'no', 'sometimes', 'yes', 'non-mammal'],
                               ['gila_monster', 'no', 'no', 'no', 'yes', 'non-mammal'],
                               ['platypus', 'no', 'no', 'no', 'yes', 'mammal'],
                               ['owl', 'no', 'yes', 'no', 'yes', 'non-mammal'],
                               ['dolphin', 'yes', 'no', 'yes', 'no', 'mammal'],
                               ['eagle', 'no', 'yes', 'no', 'yes', 'non-mammal']],
                              columns=['Name', 'GiveBirth', 'CanFly', 'LiveInWater', 'HaveLegs', 'Class'])
mammal_testing_dataset = pd.DataFrame([['uuk', 'yes', 'no', 'yes', 'no']],
                                      columns=['Name', 'GiveBirth', 'CanFly', 'LiveInWater', 'HaveLegs'])
mammal_classified_dataset = pd.DataFrame([['uuk', 'yes', 'no', 'yes', 'no', 'mammal']],
                                         columns=['Name', 'GiveBirth', 'CanFly', 'LiveInWater', 'HaveLegs',
                                                  'PredictedClass'])
# We ignore the name since we don't care
mammal_dataset = mammal_dataset.drop(columns='Name')
mammal_testing_dataset = mammal_testing_dataset.drop(columns='Name')
mammal_classified_dataset = mammal_classified_dataset.drop(columns='Name')
class_info = ('Class', ['mammal', 'non-mammal'])
feature_info = {'GiveBirth': ['no', 'yes'],
                'CanFly': ['no', 'yes'],
                'LiveInWater': ['no', 'sometimes', 'yes'],
                'HaveLegs': ['no', 'yes']}


class Task_1_Testing(unittest.TestCase):

    # This function contains one unit test for retrieving conditional probabilities.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    #

    def test1_retrieve_conditional_probability(self):
        nb = NaiveBayes(class_info, feature_info)
        nb.fit(mammal_dataset)
        class_value = 'mammal'
        feature_name = 'GiveBirth'
        feature_val = 'no'
        student_val = nb.retrieve_conditional_probability(class_value, feature_name, feature_val)
        expected = 1 / 7
        result = round_equal(student_val, expected)
        with self.subTest(
                msg='Checking conditional probability for class value ' + class_value +
                    ' feature name ' + feature_name + ' and feature value ' + feature_val):
            self.assertEqual(result, True, 'Expected ' + str(expected) + ' and got ' + str(student_val))

    # This function contains one unit test for retrieving class probabilities.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    #

    def test2_retrieve_class_probability(self):
        nb = NaiveBayes(class_info, feature_info)
        nb.fit(mammal_dataset)

        class_value = 'mammal'
        expected = 7 / 20
        student_val = nb.retrieve_class_probability(class_value)
        result = round_equal(student_val, expected)
        with self.subTest(
                msg='Checking class probability for class value ' + class_value):
            self.assertEqual(result, True, 'Expected ' + str(expected) + ' and got ' + str(student_val))

    # This function contains one unit test for naive Bayes predictions. It checks if the student output is the same
    # as the intended output, defined through mammal_classified_dataset variable at the top of the file.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    #
    def test3_predict(self):
        nb = NaiveBayes(class_info, feature_info)
        nb.fit(mammal_dataset)
        classified_data = nb.predict(mammal_testing_dataset)
        result = pd.DataFrame.equals(classified_data, mammal_classified_dataset)
        self.assertEqual(result, True,
                         'Expected ' + str(mammal_classified_dataset) + '\n and got \n' + str(classified_data))


def round_equal(nums1: float, nums2: float):
    v1 = round(float(nums1), 5)
    v2 = round(float(nums2), 5)
    if v1 != v2:
        return False
    return True


if __name__ == '__main__':
    unittest.main()
