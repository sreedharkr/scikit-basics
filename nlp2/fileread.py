try:
   fd = open('testing23.txt','a')
   fd.write('I am writing file')
   fd.write('\n')
   fd.write('My name is sreedhar')
except OSError as err:
    print("OS error: {0}".format(err))
except:
   raise Exception('file exception thrown')
finally:   
   print("closing file")
   fd.close()
   
try:
  fd2 = open('testing23.txt','r')
  for line in fd2:
    print(line)
except:
  raise Exception('could not read')
finally:
  fd2.close()
  print('closing the file')
   