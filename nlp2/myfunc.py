import nltk
from nltk import bigrams
def splitstring(str):
  tokens = str.split()
  bg = bigrams(tokens)
  print(list(bg))
  print('completed')
  
def main():
  str = 'I live in denver'
  splitstring(str)
  print('executed xxx')
  
if __name__ == "__main__":
    ret = main()
