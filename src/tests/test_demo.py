import unittest
import sys
sys.path.append('..')
from demo import *

class TestDemo(unittest.TestCase):

    def test_print_insight365(self):
        actual = "insight365 is made by guanshantech, " \
                 "please get more detail information by " \
                 "click link:www.guanshantech.com"
        result =  print_insight365_msg()
        self.assertEquals(result,actual)
