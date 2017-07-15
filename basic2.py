#append multiple items to list, create list using constructor
def test1():
    cities = ['denver','columbus','newyork']
    for a in cities:
        print(a, " ", type(a))

def test2():
    with open('mytest.txt') as f:
        try:
            while True:
                line = next(f)
                print(line, end = '')
        except StopIteration:
            print(StopIteration)
            pass

def test3():
    with open('mytest.txt') as f:
        while True:
            line = next(f, None)
            if line is None:
                break
            print(line)
        print("file", "is", "completely printed", sep = "  " )

class Person:
    list1 = ['dtree','logistic','randomF']
    def __init__(self, name, age, city):
        self.name = name
        self.age = age
        self.city = city
    def __repr__(self):
        print('in __repr__ method')
        return self.name + " "+ self.age
    def get_name(self):
        return self.name
    def get_list1(self):
        return list1

if __name__ == '__main__':
    p = Person('cluster','5','datasc')
    print(p)
    print(p.get_list1())
    
    
