""" Unit Test cases for collector """
import unittest

from application import application as testapp
from app import metadata
from app.utils import connect_db, get_table
from app.views import set_color

class TestApp(unittest.TestCase):
    """ test Flask Application """

    # executed prior to each test
    def setUp(self):
        self.client = testapp.test_client()
        self.assertEqual(testapp.debug, False)

    #executed after each test
    def tearDown(self):
        pass

    def test_main_page(self):
        """ test Home Page """
        response = self.client.get('/', follow_redirects=True)
        #print(str(response))
        self.assertEqual(response.status_code, 200)

    def test_stock(self):
        """ test Stocks dropdown """
        response = self.client.get('/stocks', follow_redirects=True)
        #print(str(response))
        self.assertEqual(response.status_code, 200)

    def test_request_params(self):
        """ test InValid Request Params """
        params = {"ticker": "UnitTest"}
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response))
        self.assertEqual(response.status_code, 400)

    def test_bad_request(self):
        """ test Request without Params """
        response = self.client.post('/process', follow_redirects=True)
        #print(str(response))
        self.assertEqual(response.status_code, 400)

    def test_get_request(self):
        """ test Get Request """
        response = self.client.get('/process', follow_redirects=True)
        #print(str(response))
        self.assertEqual(response.status_code, 405)

    def test_db(self):
        """ Check DB connection and tables """
        #connect_db(r'sqlite:///./db/StockSeasonality.db')
        with self.assertRaises(Exception):
            connect_db(r'sqlite:///./ut/StockSeasonality.db')
        with self.assertRaises(Exception):
            get_table('Test', metadata)

    def test_stock_name(self):
        """ test Valid stock name """
        params = {"stock": "20 Microns Limited"} #ImportError
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Seasonality analysis', response.data)

    def test_stock_ticker(self):
        """ test Valid stock ticker """
        params = {"stock": "20MICRONS", "ut":"Normal"}
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Seasonality analysis", response.data)

    def test_index_ticker(self):
        """ test Index ticker """
        params = {"stock": "^NSEI"}
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"", response.data)

    def test_incorrect_stock(self):
        """ test InValid stock name """
        params = {"stock": "Invalid Unit Test"}
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No stock exists", response.data)

    def test_incorrect_ticker(self):
        """ test InValid ticker name """
        params = {"stock": "Invalid"}
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No stock exists by name/ticker", response.data)

    def test_not_fetched(self):
        """ test Data not fetched for stock """
        params = {"stock": "UNITTEST", "ut":"NotFetched"}
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Data is not yet fetched for stock", response.data)

    def test_not_avl(self):
        """ test Data not available for stock """
        params = {"stock": "UNITTEST", "ut":"NotAvailable"} #e.g. COFORGE
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Data is not available for stock", response.data)

    def test_1year(self):
        """ test Less than 1 year of historical data avl. for stock """
        params = {"stock": "AARTISURF"} #e.g. AARTISURF
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"1 year", response.data)

    def test_not_clean(self):
        """ test Data not cleaned for stock """
        params = {"stock": "UNITTEST", "ut":"Raw"}
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Data is not yet cleaned for stock", response.data)

    def test_pending_analysis(self):
        """ test Analysis is pending for stock """
        params = {"stock": "UNITTEST", "ut":"Pending"}
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"is pending for stock", response.data)

    def test_no_best(self):
        """ test No Best Models for stock """
        params = {"stock": "UNITTEST", "ut":"All"}
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"all", response.data)

    def test_insufficient_data(self):
        """ test set_color method """
        params = {"stock": "21STCENMGM"}
        response = self.client.post('/process', data=params, follow_redirects=True)
        #print(str(response.data))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Data is insufficient for Seasonality analysis", response.data)

    def test_set_color(self):
        """ test set_color method """
        output = set_color("abc~1.0")
        print(output)
        self.assertIn('color', output)

if __name__ == '__main__':
    unittest.main()
    testapp.run(debug=False, port=5000)
