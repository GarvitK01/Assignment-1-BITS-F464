##################
##################
### Question 1 ###
##################
##################


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.stats as stats
from math import sqrt
import pandas as pd
import numpy as np
plt.rcParams["figure.figsize"] = (15,15)



############
### ALGO ###
############

#------------
#Reading data
#------------

pd=pd.read_csv('./dataset_FLD.csv',header=None)
data=np.array(pd)


#---------------
#Data Processing
#---------------

# Split dataset by class values
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

separated=separate_by_class(data)



#---------
#Algorithm
#---------

# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))

# Calculate the mean for each column in a dataset
def dataset_mean(dataset):
	summaries = [(mean(column)) for column in zip(*dataset)]
	del(summaries[-1])  #corresponding to target class
	return summaries

# Split dataset by class then calculate means for each row
def means_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_label, means in separated.items():
        summaries[class_label] = dataset_mean(means)
    return summaries

summary = means_by_class(data)

M1=np.array([0.0,0.0,0.0])
M2=np.array([0.0,0.0,0.0])

i=0
for label in summary:
    for row in summary[label]:
        if i<3:
            M2[i]=row
        elif i<6:
            M1[i-3]=row
        i=i+1

M=np.subtract(M1,M2,dtype=float)
M_matrix=np.matrix(M)
M_transpose=M_matrix.transpose()


neg_pts=np.matrix(separated.get(0.0))
neg_pts=neg_pts[:,:-1]
pos_pts=np.matrix(separated.get(1.0))
pos_pts=pos_pts[:,:-1]

count_neg=neg_pts.shape[0]
count_pos=pos_pts.shape[0]

proj_neg=np.zeros(shape=(count_neg,1))
proj_pos=np.zeros(shape=(count_pos,1))

SUM2=((neg_pts-M2).T@(neg_pts-M2))/count_neg
SUM1=((pos_pts-M1).T@(pos_pts-M1))/count_pos

S_w=np.add(SUM1,SUM2,dtype=float)
S_w_inverse=np.linalg.inv(S_w)

w_direction=S_w_inverse@M_transpose
w_hat = w_direction / np.linalg.norm(w_direction)
w_trans=w_hat.transpose()

proj_neg=neg_pts@w_hat
proj_pos=pos_pts@w_hat

m1=proj_pos.mean(axis=0)[0][0,0]
m2=proj_neg.mean(axis=0)[0][0,0]
std1=proj_pos.std(axis=0)[0][0,0]
std2=proj_neg.std(axis=0)[0][0,0]


#to get the intersection points of the two normal distributions
def solve(m1,m2,std1,std2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])


k=solve(m1,m2,std1,std2)[1]

print('Threshold= '+str(k))
print('\nThe unit vector w_cap= \n'+str(w_hat))
print('\nAccuracy= '+str((np.sum(np.where(proj_pos>=k, 1, 0))+np.sum(np.where(proj_neg<k, 1, 0)))*100/(proj_pos.shape[0]+proj_neg.shape[0])))


#################
##### PLOTS #####
#################

#-------------------------------------------------------------------------------
#plot of all the points, blue are the positive points, red are the negetive ones
#-------------------------------------------------------------------------------

zneg=[]
xneg=[]
yneg=[]
zpos=[]
xpos=[]
ypos=[]

ax = plt.axes(projection ="3d")

for i in range(neg_pts.shape[0]):
    xneg.append(separated.get(0.0)[i][0])
    yneg.append(separated.get(0.0)[i][1])
    zneg.append(separated.get(0.0)[i][2])
ax.scatter3D(xneg, yneg, zneg, color = "red", label='Negetive Points')

for i in range(pos_pts.shape[0]):
    xpos.append(separated.get(1.0)[i][0])
    ypos.append(separated.get(1.0)[i][1])
    zpos.append(separated.get(1.0)[i][2])
ax.scatter3D(xpos, ypos, zpos, color = "blue",label='Positive Points')

leg = ax.legend();

plt.title("Points")
plt.savefig('Points.jpg')
plt.show()

#---------------------
#Normal Distributions
#---------------------

projpos_arr=(np.array(proj_pos)).reshape(1,500)
projneg_arr=(np.array(proj_neg)).reshape(1,500)
y=np.zeros(500)

ax.set_xticks([0., 0.5, 1.])
plt.scatter(projpos_arr,y, c ="blue",  marker='|')
plt.scatter(projneg_arr,y, c ="red",  marker='|')
plt.title("Points projected on w")
plt.savefig('Points projected on w.jpg')
plt.show()

x1 = np.linspace(m1 - 5*std1, m1 + 5*std1, 100)
plt.plot(x1, stats.norm.pdf(x1, m1, std1))
plt.title("Normal distribution for positive points")
plt.savefig('Normal distribution for positive points.jpg')
plt.xlabel('w')
plt.show()

x2 = np.linspace(m2 - 5*std2, m2 + 5*std2, 100)
plt.plot(x2, stats.norm.pdf(x2, m2, std2),color='red')
plt.title("Normal distribution for negative points")
plt.xlabel('w')
plt.savefig('Normal distribution for negative points.jpg')
plt.show()

plt.plot(x2, stats.norm.pdf(x2, m2, std2),color='red')
plt.plot(x1, stats.norm.pdf(x1, m1, std1),color='blue')
x=k+y
y=np.linspace(0,2,500)
plt.plot(x,y,color='pink')
plt.title("Normal distributions and their point of intersection")
plt.xlabel('w')
plt.savefig('Normal distributions and their point of intersection.jpg')
plt.show()

#--------------------
#Classification plane
#--------------------

a,b,c,d = 0.00655686,0.01823739,-0.99981218,k

ax = plt.axes(projection ="3d")

xplane = np.linspace(-5,5,1000)
yplane = np.linspace(-1,1,1000)
X,Y = np.meshgrid(xplane,yplane)
Z = (d - a*X - b*Y) / c

ax.plot_surface(X, Y, Z,color='pink')

ax.scatter3D(xpos, ypos, zpos, color = "blue" ,label='Positive Points')
ax.scatter3D(xneg, yneg, zneg, color = "red" ,label='Negative Points')
leg = ax.legend();

plt.title("Classifier")
plt.savefig('Classifier.jpg')

plt.show()