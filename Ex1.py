# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:21:06 2017

@author: ShivamMaurya
"""

import numpy as np
import pandas as pd
from datetime import datetime as dt
import calendar
from datetime import timedelta
import math
import matplotlib.pyplot as plt
import random

def recurs_mean(X,l,n):
    if l == n-1:
        return X[l]
    
    if l==0:
        return (X[l]+recurs_mean(X,l+1,n))/n
    else:
        return (X[l]+recurs_mean(X,l+1,n))

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
    

#Part one of the excercise
print("------------Ex. 1------------")
X = np.array([4, 3, 2, 5, 4, 6, 3, 7, 4, 1, 4, 0, 6, 4, 3, 5, 2, 3, 5, 1, 4, 4, 9, 5, 4, 3, 3, 5, 2, 4,3, 6, 5, 2, 6, 2, 4, 5, 5, 1, 5, 4, 4, 2, 7, 1, 3, 3, 4, 7, 3, 4, 4, 6, 6, 3, 3, 2, 6, 1])
print(recurs_mean(X,0,len(X)))
print(X.mean())
print(X.std())

#Yelp Dataset Analysis
dataReview = pd.read_csv('../data/city/Gilbert/yelp_academic_dataset_review.csv')
dataReview = np.array(dataReview)

# Positive Lexicon
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
startDate = reviewVector[0,5].replace(day=1)
endDate = reviewVector[len(reviewVector)-1,5]
loopDate = startDate
Ti = dt(2016,1,1)

userSet = set()

numReview = 0
numReviewT = 0

posReview = 0
posReviewT = 0

negReview = 0
negReviewT = 0

meanWords = 0.0
sumWordsT = 0

meanPosWords = 0.0
meanPosWordsT = 0
 
meanNegWords = 0.0
meanNegWordsT = 0

stdDevWords = 0.0
sumSqWords = 0

stdDevPosWords = 0.0
sumSqPosWords = 0

stdDevNegWords = 0.0
sumSqNegWords = 0

plotData = []

while loopDate < Ti:
    days_in_month = calendar.monthrange(loopDate.year, loopDate.month)[1]
    loopEndDate = loopDate + timedelta(days = days_in_month)
    dateMask = (reviewVector[:,5]>loopDate) & (reviewVector[:,5] < loopEndDate)
    streamVector = reviewVector[dateMask] 
    
    numReviewT = len(streamVector)
    posReviewT,negReviewT = Review(streamVector)#,posReview,negReview)
    
    sumWordsT = sumReviewWords(streamVector)
    meanWords = (numReview*meanWords+sumWordsT)/(numReview+numReviewT)
    
    meanPosWords = (numReview*meanPosWords+np.sum(streamVector[:,7]))/(numReview+numReviewT)
    meanNegWords = (numReview*meanNegWords+np.sum(streamVector[:,8]))/(numReview+numReviewT)
    
    sumSqWords += calSumSqWords(streamVector)
    sumSqPosWords += sumSqPos(streamVector)
    sumSqNegWords += sumSqNeg(streamVector)
    
    if((numReview-1) > 0):
        stdDevWords = math.sqrt((sumSqWords-meanWords**2/numReview)/(numReview-1))
        stdDevPosWords = math.sqrt((sumSqPosWords-meanPosWords**2/numReview)/(numReview-1))
        stdDevNegWords = math.sqrt((sumSqNegWords-meanNegWords**2/numReview)/(numReview-1))
    
    numReview += numReviewT
    
    
    posReview += posReviewT
    negReview += negReviewT 
    
    userSet = newUsers(streamVector, userSet)
    plotData.append([loopDate,numReview,posReview,negReview,meanWords,meanPosWords,meanNegWords,stdDevWords,stdDevPosWords,stdDevNegWords,len(userSet)])
    
    loopDate = loopEndDate
plotData = np.array(plotData)
print("------------Ex. 2------------")
print("Time - From: ",startDate.date().isoformat()," To: ",loopDate.date().isoformat()," No of Reviews: ",numReview)
print("Positive Review: ", posReview," Negetive Reviews: ", negReview)
print("Mean Words: ", meanWords, " Mean Positive Words: ", meanPosWords, " Mean Negetive Words: ", meanNegWords)
print("Std Dev Words: ", stdDevWords, " Std Dev Pos Words: ", stdDevPosWords, " Std Dev Words: ",stdDevNegWords)
print("No of Unique Users: ", len(userSet))


print("------------Ex. 6------------")
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot(plotData[:,0],plotData[:,1],'b-',label = 'No Review')
ax1.plot(plotData[:,0],plotData[:,2],'r-',label = 'Pos Review')
ax1.plot(plotData[:,0],plotData[:,3],'g-',label = 'Neg Review')
ax1.set_title('Review (Total, Pos, Neg) / Time',fontsize=7)
ax1.legend(loc='best',fontsize=5)
ax1.set_xlabel('Time', fontsize=7)
ax1.set_ylabel('Words', fontsize=7)

ax2 = fig.add_subplot(222)
ax2.plot(plotData[:,0],plotData[:,4],'b-',label = 'Mean Words')
ax2.plot(plotData[:,0],plotData[:,7],'r-',label = 'Std Dev Words')
ax2.set_title('Mean-Std Dev Words/ Time',fontsize=7)
ax2.legend(loc='best',fontsize=5)
ax2.set_xlabel('Time', fontsize=7)
ax2.set_ylabel('Words', fontsize=7)

ax3 = fig.add_subplot(223)
ax3.plot(plotData[:,0],plotData[:,5],'b-',label = 'Mean Pos Words')
ax3.plot(plotData[:,0],plotData[:,8],'r-',label = 'Std Dev Pos Words')
ax3.set_title('Mean-Std Dev Pos Words/ Time',fontsize=7)
ax3.legend(loc='best',fontsize=5)
ax3.set_xlabel('Time', fontsize=7)
ax3.set_ylabel('Words', fontsize=7)

ax4 = fig.add_subplot(224)
ax4.plot(plotData[:,0],plotData[:,6],'b-',label = 'Mean Neg Words')
ax4.plot(plotData[:,0],plotData[:,9],'r-',label = 'Std Dev Neg Words')
ax4.set_title('Mean-Std Dev Neg Words/ Time',fontsize=7)
ax4.legend(loc='best',fontsize=5)
ax4.set_xlabel('Time', fontsize=7)
ax4.set_ylabel('Words', fontsize=7)

plt.tight_layout()
plt.show()

print("------------Ex. 5------------")
startDate = reviewVector[0,5].replace(day=1)
endDate = reviewVector[len(reviewVector)-1,5]
loopDate = startDate
Ti = dt(2016,1,1)

userSetReserve = set()

plotDataReserve = np.array(plotData[:,[0,10,10]])
days_in_month = calendar.monthrange(loopDate.year, loopDate.month)[1]
loopEndDate = loopDate + timedelta(days = days_in_month)
dateMask = (reviewVector[:,5]>loopDate) & (reviewVector[:,5] < loopEndDate)
reservVector = reviewVector[dateMask] 

lenResVector = len(reservVector)
lenReviewVector = len(reviewVector)

randIndex = 0
plotIndex = 0
for i in range(lenResVector-1, lenReviewVector-1):
    randIndex = random.randrange(0,lenResVector,1)
    reservVector[randIndex,:] = reviewVector[i,:]#Replace row
    userSetReserve.add(reviewVector[i,1])
    plotIndex = np.where(plotData[:,0]==reviewVector[i,0])
    if(plotIndex):
        plotDataReserve[plotIndex,2]=len(userSetReserve)
print("Unique Users Reserve: ",plotDataReserve[len(plotDataReserve)-1,2])
f=plt.figure(2)
plt.plot(plotDataReserve[:,0],plotDataReserve[:,1],'b-',label='Users Split')
plt.plot(plotDataReserve[:,0],plotDataReserve[:,2],'g-',label='Users Reserve')
plt.xlabel("Time")
plt.ylabel("Unique User")
plt.legend(loc='best')
plt.show()
    