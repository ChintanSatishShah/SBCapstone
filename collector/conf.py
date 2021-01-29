"""Configs"""

class CollectorConfig:
    """Set Collector config variables."""

    #Path on Shared drive to store collected raw data
    RAWDATA_DIRECTORY = r'./data/rawdata/'
    #Log file path
    LOG_DIRECTORY = r'./logs/'
    #critical-50,error-40,warning-30,info-20,debug-10,notset-0
    LOG_LEVEL = 20
    #Throttling no. of stocks for which raw data is fetched and saved, default = 2000
    STOCKS_PER_CYCLE = 5
