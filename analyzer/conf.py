"""Configs"""

class AnalyzerConfig:
    """Set Collector config variables."""

    #Path on Shared drive to store collected raw data
    DIR_RAWDATA = r'./data/rawdata/'
    DIR_MOD = r'./data/moddata/'
    #Log file path
    LOG_DIRECTORY = r'./logs/'
    #critical-50,error-40,warning-30,info-20,debug-10,notset-0
    LOG_LEVEL = 20
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    #Path on Shared drive to store output
    OUTPUT_DIRECTORY = r'./output/'
    WEBAPP_DIRECTORY = r'./app/static/'
    #Directory to store models
    DIR_MODEL = r'model/'
    #Directory to store csv results
    DIR_RESULT = r'csv/'
    #Directory to store images
    DIR_IMAGE = r'img/'
