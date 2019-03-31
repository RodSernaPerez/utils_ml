import unittest

import numpy as np

from utils_ml.src.data_processing import data_processing_utils


class TestDataProcessingUtils(unittest.TestCase):
    MSG = "Data processing utils:: UNITTEST:: "

    def test_convert_to_one_hot_vectors_list_with_ints(self):
        msg = self.MSG + "convert_to_one_hot_vectors_list_with_ints:: converts to one hot vectors a list of ints"

        input_list = [0, 2, 1, 2]

        expected_output = [[1, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0],
                           [0, 0, 1]]

        result = data_processing_utils.convert_to_one_hot_vectors(input_list)

        self.assertTrue(isinstance(result, list),
                        msg + "::Error: type of output is not list")

        self.assertTrue(
            np.array_equal(
                np.asarray(result),
                np.asarray(expected_output)),
            msg + "::Error: result is not right")
        print(msg + "::OK")

    def test_convert_to_one_hot_vectors_list_with_strings(self):
        msg = self.MSG + "convert_to_one_hot_vectors_list_with_strings:: converts to one hot vectors a list of strings"

        input_list = ["1", "3", "2", "3"]

        expected_output = [[1, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0],
                           [0, 0, 1]]

        result = data_processing_utils.convert_to_one_hot_vectors(input_list)

        self.assertTrue(isinstance(result, list),
                        msg + "::Error: type of output is not list")

        self.assertTrue(
            np.array_equal(
                np.asarray(result),
                np.asarray(expected_output)),
            msg + "::Error: result is not right")
        print(msg + "::OK")

    def test_convert_to_one_hot_vectors_numpy_with_ints(self):
        msg = self.MSG + "convert_to_one_hot_vectors_numpy_with_ints:: converts to one hot vectors a numpy of ints"

        input_list = np.asarray([0, 2, 1, 2])

        expected_output = [[1, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0],
                           [0, 0, 1]]

        result = data_processing_utils.convert_to_one_hot_vectors(input_list)

        self.assertTrue(isinstance(result, np.ndarray),
                        msg + "::Error: type of output is not numpy")

        self.assertTrue(
            np.array_equal(
                result,
                expected_output),
            msg + "::Error: result is not right")
        print(msg + "::OK")

    def test_convert_to_one_hot_vectors_numpy_with_strings(self):
        msg = self.MSG + \
              "convert_to_one_hot_vectors_numpy_with_strings:: converts to one hot vectors a numpy of strings"

        input_list = np.asarray(["1", "3", "2", "3"])

        expected_output = np.asarray([[1, 0, 0],
                                      [0, 0, 1],
                                      [0, 1, 0],
                                      [0, 0, 1]])

        result = data_processing_utils.convert_to_one_hot_vectors(input_list)

        self.assertTrue(isinstance(result, np.ndarray),
                        msg + "::Error: type of output is not numpy")

        self.assertTrue(
            np.array_equal(
                result,
                expected_output),
            msg + "::Error: result is not right")
        print(msg + "::OK")


if __name__ == '__main__':
    unittest.main()
