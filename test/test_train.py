import unittest
from expred.train import main

class TestTrain(unittest.TestCase):
    def test_runs_for_one_epoch(self):
        # todo fix path / data problem
        args = [
            "--data_dir", "/home/mreimer/datasets/eraser/movies_debug",
            "--output_dir", "output/",
            "--conf", "test_params/movies_expred.json",
            "--batch_size", "4"]
        main(args)


if __name__ == '__main__':
    unittest.main()
