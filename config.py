"""Stores Flask application configurations"""

#from os import path, environ
#from dotenv import load_dotenv

#basedir = path.abspath(path.dirname(__file__))
#load_dotenv(path.join(basedir, '.env'))

class Config:
    """Set Flask config variables."""

    FLASK_ENV = 'development'
    TESTING = True
    #SECRET_KEY = environ.get('SECRET_KEY')
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'

    # Database
    #SQLALCHEMY_DATABASE_URI = environ.get('SQLALCHEMY_DATABASE_URI')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///../db/StockSeasonality.db'
    DATABASE_URI = r'sqlite:///./db/StockSeasonality.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False #For Dev, keep True

    #Log file path
    LOG_DIRECTORY = r'./logs/'
    #critical-50,error-40,warning-30,info-20,debug-10,notset-0
    LOG_LEVEL = 20
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # AWS Secrets
    #AWS_SECRET_KEY = environ.get('AWS_SECRET_KEY')
    #AWS_KEY_ID = environ.get('AWS_KEY_ID')

    #Path on Shared drive to store output
    OUTPUT_DIRECTORY = r'./output/'
    #Directory to store models
    DIR_MODEL = r'model/'
    #Directory to store csv results
    DIR_RESULT = r'csv/'
    #Directory to store images
    DIR_IMAGE = r'img/'
