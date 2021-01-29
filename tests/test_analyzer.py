""" Unit Test cases for Analyzer """
import unittest
from datetime import datetime

from analyzer.utils import create_dir, connect_db, get_table
from analyzer.analyze import metadata, start_analysis, check_seasonality

class TestAnalyzer(unittest.TestCase):
    """ test Analyzer """

    def test_db(self):
        """ Check DB connection and tables """
        #connect_db(r'sqlite:///./db/StockSeasonality.db')
        with self.assertRaises(Exception):
            connect_db(r'sqlite:///./ut/StockSeasonality.db')
        with self.assertRaises(Exception):
            get_table('Test', metadata)

    def test_create_dir(self):
        """ test collect """
        self.assertIsNone(create_dir(r'./data/rawdata/'))
        self.assertIsNone(create_dir(r'./data/ut' + str(datetime.now().timestamp()) + r'/'))
        self.assertIsNone(create_dir(r'./test/rawdata/'))

    def test_start_analysis(self):
        """ test analyzer main code """
        self.assertIsNone(start_analysis(True))

    def test_check_seasonality(self):
        """ test exception in check_seasonality """
        with self.assertRaises(Exception):
            check_seasonality('abc', True)

if __name__ == '__main__':
    unittest.main()
