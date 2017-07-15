'''
sys.path
import os
cwd = os.getcwd()
sys.path.append('path')
reload(hello_mod)
'''

def calc(a):
  print __name__
  print 3*a
  
