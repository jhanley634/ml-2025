"""
Brief demo of how to import from a sandbox module.

Run with `$ make test`
"""

import unittest

from sandbox.jhanley634.product import product


class ProductTest(unittest.TestCase):

    def test_product(self) -> None:
        self.assertEqual(24, product(1, 2, 3, 4))
        self.assertEqual(24, product(4, 3, 2, 1.0))
        self.assertEqual(24, product(*range(1, 5)))
