# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:07:54 2017

@author: ShivamMaurya
"""
'''
Excercise 2: Consider the scenario of Exercise 1, Task 2. We still consider the 
reviews as a stream with the timepoint being the 1st of each month. Let the 
window size be 3 months. Calculate
a. For timepoint ti, the number of reviews in the window
b. For timepoint ti, the number of positive/negative reviews in the window
c. For timepoint ti, the number of users in the window
Use a Sliding window and Landmark window for the above calculations.
'''
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
#%matplotlib inline

def positiveWords(text,pos_words):
    posW = 0
    for w in text:
        if pos_words.issuperset({w.lower()}):
            posW+=1
    return posW

def negetiveWords(text,neg_words):
    negW = 0
    for w in text:
        if neg_words.issuperset({w.lower()}):
            negW+=1
    return negW    

def Review(streamVector):
    posReview = 0
    negReview = 0
    for instance in streamVector:
        if(instance[3]>2):
            posReview+=1
        else:
            negReview+=1
    return posReview,negReview

def sumReviewWords(streamVector):
    sumWords = 0
    for instance in streamVector:
        sumWords += instance[6]
        
    return sumWords

def calSumSqWords(stremVector):
    sumSquare = 0
    for instance in streamVector:
        sumSquare += instance[6]**2
    return sumSquare
        
def sumSqPos(streamVector):
    sumSquare = 0
    for instance in streamVector:
        sumSquare+=instance[7]**2
    return sumSquare

def sumSqNeg(streamVector):
    sumSquare = 0
    for instance in streamVector:
        sumSquare += instance[8]**2
    return sumSquare

def newUsers(streamVector,userSet):
    for instance in streamVector:
        userSet.add(instance[1])
    return userSet
    
#Yelp Dataset Analysis
dataReview = pd.read_csv('../data/city/Gilbert/yelp_academic_dataset_review.csv')
dataReview = np.array(dataReview)

#Positive Lexicon
pos_words = set()
for words in open('../data/opinion lexicon/positive-words.txt', 'r').readlines()[35:]:
    pos_words.add(words.rstrip())

#Negative Lexicon
neg_words = set()
for words in open('../data/opinion lexicon/negative-words.txt', 'r').readlines()[35:]:
    neg_words.add(words.rstrip())

reviewVector = []
reviewLine = [0,0,0,0,0,0,0,0]
for line in dataReview:
    text = line[5][3:len(line[5])-2].split(' ')
    date = dt(int(line[2][2:6]),int(line[2][7:9]),int(line[2][10:len(line[2])-1]))
    busId = line[6][2:len(line[6])-1]
    revId = line[4][2:len(line[3])-1]
    userId = line[3][2:len(line[3])-1]
    reviewVector.append(np.array([revId,userId,busId,line[8],line[5][3:len(line[5])-2],date,len(text), positiveWords(text,pos_words),negetiveWords(text,neg_words)]))
    
reviewVector = np.array(reviewVector)
reviewVector = reviewVector[reviewVector[:,5].argsort()]

#Stremify the date
#Find first date in the record and then use it to take first of month the date comes from
startDate = dt(2010,1,1)#reviewVector[0,5].replace(day=1)
endDate = reviewVector[len(reviewVector)-1,5]
loopDate = startDate
Ti = dt(2016,1,1)

userSet = set()
userSetT=set()

numReview = 0
numReviewT = 0

posReview = 0
posReviewT = 0

negReview = 0
negReviewT = 0

meanWords = 0
plotData = []


#Sliding Windows

while loopDate < Ti:
    loopEndDate = loopDate + timedelta(days = 90)
    dateMask = (reviewVector[:,5]>loopDate) & (reviewVector[:,5] < loopEndDate)
    streamVector = reviewVector[dateMask] 
    
    numReviewT = len(streamVector)
    posReviewT,negReviewT = Review(streamVector)#,posReview,negReview)
    
    sumWordsT = sumReviewWords(streamVector)
    meanWords = (numReview*meanWords+sumWordsT)/(numReview+numReviewT)
    
    numReview += numReviewT
    
    
    posReview += posReviewT
    negReview += negReviewT 
    
    
    userSet = newUsers(streamVector, userSet)
    userSetT.clear()
    userSetT = newUsers(streamVector, userSetT)
    #print(loopDate.date(),numReviewT,posReviewT,negReviewT,len(userSetT))
    plotData.append([loopDate.date(),numReviewT,posReviewT,negReviewT,len(userSetT)])
    
    loopDate = loopEndDate
plotData = np.array(plotData)
print("------------Ex. 2------------")
print("Time - From: ",loopDate.date().isoformat()," To: ",loopDate.date().isoformat()," No of Reviews: ",numReviewT)
print("Positive Review: ", posReviewT," Negetive Reviews: ", negReviewT)
print("Total No of Unique Users: ", len(userSet))


ax = plt.subplot(111)
dates = date2num(plotData[:,0])
ax.bar(dates, plotData[:,1],width=35,color='b',align='center',label='Reviews')
ax.bar(dates, plotData[:,2],width=35,color='g',align='center',label='Pos Review')
ax.bar(dates, plotData[:,3],width=35,color='r',align='center',label='Neg Review')
ax.bar(dates+35, plotData[:,4],width=35,color='y',align='center',label='User')
ax.xaxis_date()
ax.set_xticks(plotData[:,0])
ax.set_xticklabels(plotData[:,0],rotation=90, size=7)
ax.autoscale(tight=True)
ax.legend(loc='best')
title = "Sliding Window(90 Days): Reviews-User from "+str(startDate.date())+" to "+str(loopEndDate.date())
ax.set_title(title,size=7)
plt.show()
plt.close()

#Landmark window
#If no neg review more than 500
startDate = dt(2010,1,1)#reviewVector[0,5].replace(day=1)
endDate = reviewVector[len(reviewVector)-1,5]
loopDate = startDate
Ti = dt(2016,1,1)

userSet = set()
userSetT=set()

numReview = 0
numReviewT = 0

posReview = 0
posReviewT = 0

negReview = 0
negReviewT = 0

meanWords = 0
plotData = []

LandmarkWindowData = reviewVector[np.where(reviewVector[:,5]>startDate)]
for line in LandmarkWindowData:
    numReviewT +=1
    posReviewT += (0,1)[line[3]>2]
    negReviewT += (0,1)[line[3]<=2]
    userSetT.add(line[1])
    if negReviewT == 500:
        #print(line[5].date(),numReviewT,posReviewT,negReviewT,len(userSetT))
        plotData.append([line[5].date(),numReviewT,posReviewT,negReviewT,len(userSetT)])
        userSetT=set()
        numReviewT = 0
        negReviewT = 0
        posReviewT = 0

plotData=np.array(plotData)
dates = date2num(plotData[:,0])
ax = plt.subplot(111)
ax.plot_date(dates, plotData[:,1],fmt='b-',label='Reviews')
ax.plot_date(dates, plotData[:,2],fmt='g-',label='Pos Review')
ax.plot_date(dates, plotData[:,4],fmt='y-',label='User')

ax.set_xticks(plotData[:,0])
ax.set_xticklabels(plotData[:,0],rotation=90, size=6)
ax.autoscale(tight=True)

ax.legend(loc='best')
ax.grid(axis='both')

title = "Landmark Window(Neg Review=500): Reviews-User from "+str(startDate.date())+" to "+str(loopEndDate.date())

ax.set_title(title, size=7)

plt.show()
plt.close()

#Damped Window
startDate = dt(2016,1,1)
endDate = dt(2017,1,2)
lmd = 1 #Lambda value
userSet = set()
userSetT=set()
lenUser = 0
numReview = 0
numReviewT = 0

posReview = 0
posReviewT = 0

negReview = 0
negReviewT = 0

meanWords = 0
plotData = []
f_t = 0

DampedWindowData = reviewVector[np.where(reviewVector[:,5]>startDate)]#
DampedWindowData = DampedWindowData[np.where(DampedWindowData[:,5]<endDate)]
for line in DampedWindowData:
    f_t = 2**(-lmd*(endDate.date()-line[5].date()).days)
    numReview += f_t
    posReviewT += (0,f_t)[line[3]>2]
    negReviewT += (0,f_t)[line[3]<=2]
    lenUser += (0,f_t)[userSetT.issuperset({line[1]})]
    userSetT.add(line[1])
    plotData.append([line[5].date(),numReviewT,posReviewT,negReviewT,lenUser])
    
plotData=np.array(plotData)
#dates = date2num(plotData[:,0])
ax = plt.subplot(111)
ax.plot_date(plotData[:,0], plotData[:,1],fmt='b-',label='Reviews')
ax.plot_date(plotData[:,0], plotData[:,2],fmt='g-',label='Pos Review')
ax.plot_date(plotData[:,0], plotData[:,3],fmt='r-',label='Neg Review')
ax.plot_date(plotData[:,0], plotData[:,4],fmt='y-',label='User')
ax.autoscale(tight=True)

ax.set_xticklabels(ax.get_xticks(),rotation=90, size=6)

ax.legend(loc='best')
ax.grid(axis='both')

title = "Damping Window(lambda="+str(lmd)+"): Reviews-User from "+str(startDate.date())+" to "+str(endDate.date())

ax.set_title(title, size=7)

plt.show()
plt.close()
