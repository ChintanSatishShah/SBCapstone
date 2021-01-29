""" serve as our entry point """
from app import create_app

# pylint: disable=invalid-name
application = MYAPP = create_app()

if __name__ == "__main__":
    #MYAPP.run(debug=True, port=5000, threaded=True)
    MYAPP.run(port=5000, threaded=True)
    #MYAPP.run(host='0.0.0.0')
