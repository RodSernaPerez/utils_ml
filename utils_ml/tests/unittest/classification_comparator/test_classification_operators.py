import unittest

import numpy as np

from utils_ml.src.classification_comparator.classification_comparator import ClassificationComparator


class TestClassificationComparator(unittest.TestCase):
    MSG = "Classification Comparator:: UNITTEST:: "

    def test_run_processes(self):
        msg = self.MSG + "run_processes:: runs all the processes"

        cc = ClassificationComparator()

        number_of_samples: int = 100
        number_of_variables: int = 200

        input_variables = np.random.rand(number_of_samples, number_of_variables)
        targets = np.round(np.random.rand(number_of_samples))

        cc.run(input_variables, targets)

        print(msg + "::OK")


if __name__ == '__main__':
    unittest.main()
