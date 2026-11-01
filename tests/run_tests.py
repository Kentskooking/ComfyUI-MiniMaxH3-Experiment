import sys
import unittest
from pathlib import Path

import torch


if __name__ == "__main__":
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
