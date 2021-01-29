""" Unit Test cases for collector """

import unittest
from datetime import datetime

from collector.collect import tracker, const, metadata, start_collector, connect_db, get_table
from collector.collect import create_dir, get_stock_list, get_history_data, get_pending_stocks

class TestCollect(unittest.TestCase):
    """ test collect """

    def test_db(self):
        """ Check DB connection and tables """
        #connect_db(r'sqlite:///./db/StockSeasonality.db')
        with self.assertRaises(Exception):
            connect_db(r'sqlite:///./ut/StockSeasonality.db')
        with self.assertRaises(Exception):
            get_table('Test', metadata)

    def test_get_stock_list(self):
        """ test collect """
        self.assertTrue(get_stock_list(True) > 0)

    def test_get_pending_stocks(self):
        """ test collect """
        self.assertIn(len(get_pending_stocks(2)), [0, 2])

    def test_get_history_data(self):
        """ test collect """
        valid = tracker.select((tracker.c.ForecastYear == const.CURRENT_YEAR)
                               & (tracker.c.ListingTicker == 'VOLTAS')
                               ).limit(1).execute().fetchall()
        self.assertIsNone(get_history_data(valid, 1))
        #RemoteDataError simulation
        self.assertIsNone(get_history_data(valid, 2))
        #ValueError simulation
        self.assertIsNone(get_history_data(valid, 3))

    def test_index(self):
        """ test collector for Index """
        index = tracker.select((tracker.c.ForecastYear == const.CURRENT_YEAR)
                               & (tracker.c.ListingTicker == '^NSEI')
                               ).execute().fetchall()
        self.assertIsNone(get_history_data(index, 1))

    def test_invalid_stock(self):
        """ test collector for Incorrect stock """
        invalid = tracker.select((tracker.c.ForecastYear == const.CURRENT_YEAR)
                                 & (tracker.c.ListingTicker == 'COFORGE')
                                 ).execute().fetchall()
        self.assertIsNone(get_history_data(invalid, 1))

    def test_create_dir(self):
        """ test collect """
        self.assertIsNone(create_dir(r'./data/rawdata/'))
        self.assertIsNone(create_dir(r'./data/ut' + str(datetime.now().timestamp()) + r'/'))
        self.assertIsNone(create_dir(r'./test/rawdata/'))

    def test_start_collector(self):
        """ test collect """
        self.assertIsNone(start_collector(True))

if __name__ == '__main__':
    unittest.main()
