class MyIterator:

    def __init__(self, data):
        self.arr = data
        self.index = len(self.arr)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        else:
            self.index = self.index -1
            return self.arr[self.index]
    def var_args(self, *args):
        print(args)
        print(type(args))
    
