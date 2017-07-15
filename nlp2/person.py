import customer
from customer import Customer
class Person(Customer):


	def __init__(self, name, balance,city):
		self.name = name
		self.balance = balance
		self.city=city
		
	def print_data(self):
		print self.name+" " + str(self.balance) + " "+ self.city
		
def client():
   p = Person('sree',5000,'denver')
   p.print_data()