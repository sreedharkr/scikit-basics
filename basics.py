
def starts(elm):
    if len(elm) > 0:
        return elm[0] == 'a'

    
c4 = ['aa','bb','aa','aa','bb','cc']
f = filter(starts, c4)
print(list(f))

f2  = filter(lambda x : x.startswith('bb'), c4)



np.arange(10,30,2).reshape(2,5)
np.linspace(2,3,num=5)
np.random.random((3,2))
m1  = np.zeros((3,2),dtype=float)
m1.sum(axis=0) #rows
m1.min(axis=0)
m1[m1 > 4]

m1[np.ix_([0,1],[2,3])]
#types of indexing
df8 = pd.DataFrame(np.random.randn(10, 5), columns = [list('abcde')],index= list('abcdefghij'))
#select column
df8['a']
df8['a','b']
df8.loc[('a','b'),('d','e')]
aa = df8['c'] == 'test'
df8[~aa]
df8[aa]
df8.iloc[:, :3] # first 3 columns
df8[df8['c'] == 'test']
df8[df8['c'] != 'test']
df8[ (df8['c'] != 'test') | (df8['d'] > 0) ]
df8[(df8['c'] == 'test') | (df8['e'] == 'test')]

'test' in df8['a'].tolist()

col = []
for a in df8.columns.values:
    if('test' in df8[a].tolist()):
        col.append(a)
        
# get all row indexe with na values
# pd.notnull() pd.isnull()
pd.isnull(df8).any(1)

def mymap(x):
    if (x == 0):
        return "class1"
    if(x == 1):
        return "class2"
    if(x == 2):
        return "class3"


    



