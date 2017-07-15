################# 2-sample t-test for independence #############
hardf = pd.read_table('dataset-har.txt', sep = ';')
hardf['class'].value_counts()
sittingx1 = hardf.loc[ hardf['class'] == 'sitting',['x1'] ]
sittingx1.shape
walkingx1 = hardf.loc[hardf['class'] == 'walking', 'x1']
walkingx1.shape
#This is a two-sided test for the null hypothesis that 2 independent samples
#have identical average (expected) values.

stats.ttest_ind(sittingx1,walkingx1) # less than 0.05

#sampling from sittingx1
indi = rng1.choice(2, len(sittingx1))
indi.tolist().count(1)
samp1 = sittingx1[indi == 0 ]
samp2 = sittingx1[indi == 1]
stats.ttest_ind(samp1,samp2)


#2#The one-way ANOVA ##################
#tests the null hypothesis that two or more groups have
    #the same population mean.
np.random.seed(12)
import numpy as np
rng2 = np.random.RandomState(12)
races =   ["asian","black","hispanic","other","white"]
# Generate random data
voter_race = rng2.choice(a= races,
                              p = [0.05, 0.15 ,0.25, 0.05, 0.5],
                              size=1000)
voter_age = stats.poisson.rvs(loc=18,
                              mu=30,
                              size=1000)
# Group age data by race
voter_frame = pd.DataFrame({"race":voter_race,"age":voter_age})
voter_frame['race'].value_counts()
groups = voter_frame.groupby("race").groups

# Etract individual groups
asian = voter_age[groups["asian"]]

asian_indices = groups['asian']
vf = voter_frame.loc[asian_indices,]
vf.shape

black = voter_age[groups["black"]]
hispanic = voter_age[groups["hispanic"]]
other = voter_age[groups["other"]]
white = voter_age[groups["white"]]

stats.f_oneway(asian, black, hispanic, other, white)


hardf = pd.read_table('dataset-har.txt', sep = ';')
x1 = hardf['x1']
y1 = hardf['y1']
z1 = hardf['z1']
stats.f_oneway(x1,y1,z1)

##### chi-square test ###################
table1 = pd.crosstab(hardf['class'],hardf.gender)
stats.chi2_contingency(table1)
