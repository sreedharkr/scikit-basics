import pandas as pd
import numpy

class MyClass:
    """A simple example class"""
    _city = "Denver" 
    def print_city(self):
        print(self._city)

    def loops_test(self,start, stop, step = 2):
        list1 = numpy.arange(start,stop,step)
        for a in list1:
            print(a)

class Machine:

    concept = "basics of scikit and python"
    total = 0
    
    """ First classon Machine Learning """
    def iris_info(self,data):
        print(dir(data))
        temp = numpy.c_[data.data, data.target]
        fnames = ['Sepal Lemgth','Sepal Width','Petal Length','Petal Width','Species']
        df = pd.DataFrame(data = temp, columns = fnames )
        print(df.info())
        print(df.sample())

    def test(self):
        while True:
            print('Hello')
            self.total = self.total + 1
            if self.total > 10:
                break
class Neural(Machine):
    def load_data(self):
        print('loading data')
        
