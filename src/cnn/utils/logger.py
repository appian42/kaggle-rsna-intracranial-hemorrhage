import sys
import os
from datetime import datetime
import logging


class Logger(object):

    def __init__(self):
        self.logger = logging.getLogger()        
        self.logger.setLevel(logging.DEBUG)
        #self.logger.addHandler(logging.StreamHandler())

    def setup(self, dirname, name):

        os.makedirs(dirname, exist_ok=True)
    
        path = f'{dirname}/{name}.log'
        file_handler = logging.FileHandler(path, 'a')

        self.logger.addHandler(file_handler)

        log('')
        log('----- %s -----' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        log(' '.join(sys.argv))
        log('logpath: %s' % path)


def log(msg):
    print(msg)
    logger.logger.info(msg)


logger = Logger()
