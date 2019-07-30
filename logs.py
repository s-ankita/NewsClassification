# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:00:07 2019

@author: dbda
"""

import logging
#import mylib

def main():
    pass
    #logging.info('Started NEW EXECUTION')
    #mylib.do_something()
    #logging.info('Finished EXECUTION')

if __name__ == '__main__':
    logging.basicConfig(filename='myapp.log', format='%(asctime)s %(message)s',level=logging.INFO,)
    main()