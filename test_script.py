#!/usr/bin/env python

if __name__ == '__main__':

    from test.commands import *

    test_latest() # good
    test_random_methods() # good   

    test_lr_normalizer() # good
    test_predict() # good
    test_reducers() # good
    test_templates() # good

    test_autom8() # good
    
    scan_object = test_scan() # good
    test_analyze(scan_object) # good
    # test_rest(scan_object) # bad

    print("\n All tests successfully completed :) Good work. \n ")






