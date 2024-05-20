import unittest
from solution import result


class TestResultFunction(unittest.TestCase):
    def test_white_advance_move(self):
        initial_state = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]
        action = ["advance", 2, 2]
        expected_state = [1, -1, -1, -1, 0, 0, 1, 1, 1, 0]
        self.assertEqual(result(initial_state, action), expected_state)

    def test_white_captureLeft_move(self):
        initial_state = [0, -1, 0, 0, 0, 1, 0, 0, 0, 0]
        action = ["captureLeft", 1, 1]
        expected_state = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(result(initial_state, action), expected_state)

    def test_white_captureRight_move(self):
        initial_state = [0, 0, 0, 0, 0, -1, 0, 1, 0, 0]
        action = ["captureRight", 2, 0]
        expected_state = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        self.assertEqual(result(initial_state, action), expected_state)

    def test_black_advance_move(self):
        initial_state = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0]
        action = ["advance", 1, 1]
        expected_state = [0, 0, 0, 0, 0, 0, 0, 0, -1, 0]
        self.assertEqual(result(initial_state, action), expected_state)

    def test_black_captureLeft_move(self):
        initial_state = [1, -1, 0, -1, 0, -1, 1, 1, 1, 1]
        action = ["captureLeft", 1, 1]
        expected_state = [0, -1, 0, -1, 0, 0, 1, -1, 1, 1]
        self.assertEqual(result(initial_state, action), expected_state)

    def test_black_captureRight_move(self):
        initial_state = [1, -1, 0, -1, 0, 1, 0, 1, 0, 1]
        action = ["captureRight", 0, 0]
        expected_state = [0, 0, 0, -1, 0, -1, 0, 1, 0, 1]
        self.assertEqual(result(initial_state, action), expected_state)


if __name__ == "__main__":
    unittest.main()
