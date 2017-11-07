# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt
from math import *
from numpy import nanmean


def reduceListIntervals(deltaInit,lengthInterval,liste):
	'''Enter a list of n bins (n is maximal). Give a length of intervals. 
	It returns a shorter list that sums up juxtaposed values

	liste: The initial long list
	lengthInterval: The desired length of the intervals to be obtained
	deltaInit: The point at which to start merging (usually 0)'''

	maxNumberOfBins = len(liste)
	mergedList = []
	i = 0
	while ((i+1)*lengthInterval + deltaInit) <=  maxNumberOfBins:
		mergedList.append(np.sum(liste[(i*lengthInterval + deltaInit): ((i+1)*lengthInterval + deltaInit)]))
		i +=1
	if len(liste[(i*lengthInterval + deltaInit): maxNumberOfBins])> 0:
		mergedList.append(np.sum(liste[(i*lengthInterval + deltaInit): maxNumberOfBins]))
	return mergedList





def reduceListNumberBins(deltaInit,requestedB,liste):
	'''Enter a list of n bins (n is maximal). Give a number of bins. 
	It returns the merged list that sums up juxtaposed values

	liste: The initial long list
	requestedB: The desired number of bins
	deltaInit: The point at which to start merging (usually 0)'''

	listIntegers = np.array([float(i) for i in range(len(liste))])
	widthInterval = float((len(liste) - deltaInit)/requestedB)
	mergedList = [0] * requestedB
	for  i in range(len(mergedList)-1):
		a = deltaInit + i * widthInterval
		b = deltaInit + (i+1) * widthInterval
		indexes = np.where((a<=listIntegers) & (listIntegers <b))[0]
		mergedList[i] = np.sum([liste[u] for u in indexes])
	return mergedList, widthInterval
	



def initializePandPi(K,B,data,epsilonForInitialization = 0.1):

	# data must be a list
	if type(data) is not list:   
		data = data.tolist()

	#sum of pi must be =  1 
	Pi = list(np.random.dirichlet(np.ones(K),size=1)[0])   
		   
	# the process for initializing P
	P = []
	# the strategy for initializing correctly P: (1-2epsilonForInitialization) * H_qlcq + epsilonForInitialization * H_moyen + epsilonForInitialization * unif
	H_moyen = nanmean(data, axis = 0)
	#print("Data: ", data)
	#print("Average histogram: ", H_moyen)

	for k in range(K):
		H_l = random.choice(data)
		T1 = [(1-2*epsilonForInitialization)*u for u in H_l]
		#print ("Term 1 for P cluster " , k, ": ", T1)
		
		
		
		T2 = [epsilonForInitialization * u for u in H_moyen]
		#print ("Term 2 for P cluster " , k, ": ", T2)
		
		T3 = [epsilonForInitialization/B]*len(data[0])
		#print ("Term 3 for P cluster " , k, ": ", T3)
	
		P_k = [(T1[u] + T2[u] + T3[u]) for u in range(len(data[0]))]

		# renormalize in order to have a proba distribution
		P_k = [u/sum(P_k) for u in P_k]
		P.append(P_k)

	#we need to ensure that P and Pi are not too low (the likelihood would explode)    
	P = np.maximum(P, 10**(-100))
	Pi = np.maximum(Pi, 10**(-100))
		
	return P, Pi

 
def logMultinomialDistributionFunction(data_l,P_k):
	return np.dot(np.log(P_k),  data_l)

def logMultinomialDistributionFunctionK(data,P_k):
	return np.sum(np.dot(  data , np.log(P_k)))



def logLikelihood(K, L, B, P, Pi, data):
	"""
	Calculate the log likelihood using current responsibility. 
	"""
	print("Compute likelihood using current responsibility...")
	print("Number of bins:  ", B)

	logScore = 0.0
	
	for l in range(L):
		term_l = 0
		
		#Now we include a readaptation in the computation of the log likelihood, because it can explode numerically

		if K == 1:
			term_l += log(Pi[0])+logMultinomialDistributionFunction(data[l],P[0])

		else:

			delta = []
			for k in range(K):
				delta.append(log(Pi[k])+logMultinomialDistributionFunction(data[l],P[k]))

			deltaMax = max(delta)
			kMax = np.argmax(delta)
			
			term_l += deltaMax + log(1 + sum([exp(v - deltaMax) for v in delta[:kMax] + delta[kMax+1:]])) + log(sum([exp(u - deltaMax) for u in delta])) 
            
		logScore += term_l
		#print logScore
	
	return logScore
		
		
def expectationStep(data,K,L,B,P,Pi):
	''' Evaluate the responsibility, given parameter values P and Pi'''

	print("E-step ...")
	print("Number of clusters: ", K, "/ Number of bins: ", B)
		
	R = np.zeros((L,K))

	for l in range(L):
		#print [logMultinomialDistributionFunction(data[l],P[k]) for k in range(K)]
		lambdaHat = [log(Pi[k]) + logMultinomialDistributionFunction(data[l],P[k]) for k in range(K)]
		# proba = exp (real theoretical term - compensation term in order for it to not explode) 




		lambdaMax = max(lambdaHat)
		newLambda = [lambdaHat[k] - lambdaMax for k in range(K)]
		probabilite = [(exp(1 * newLambda[u])) for u in range(K)]
		renormalization = sum(probabilite)

		R[l] = [probabilite[k]/renormalization for k in range(K)]
			
			
	return R
		
		
def maximizationStep(R, data, K, L, B):
	'''Given the a posteriori matrix R, reestimate new parameters P and Pi'''
 
	print("M-step...")
	print("Number of clusters: ", K, "/ Number of bins: ", B)
		
	newPi = np.zeros((K,1))
	sumOnL = np.sum(R, axis = 0)
	for k in range(K):
		newPi[k][0] = np.maximum( sumOnL[k], 10**(-2))
        
	normalizePi = sum(newPi)
	newPi = newPi / normalizePi
    
    
	newP = np.zeros((K,B))
	Rtranspose = R.transpose()
	matrixA  = np.dot(Rtranspose,data)

	renormalizationA = np.sum(matrixA, axis = 1)
	#print("axis = 1", renormalizationA)
    
	for k in range(K):       
		for j in range(B):
			#print(matrixA[k][j])
			newP[k][j] = max(matrixA[k][j]/renormalizationA[k], 10**(-5))                 
	return newP,newPi

	
	
def expectationMaximisation(data, K, L, B, Pinit, PiInit, thresholdConvergence, epsilonForInitialization,  maxIterations = 100):
	'''
	Enter some data of dimension L x B. Apply the EM algorithm in order to get the parameters of the multinomial mixture.

	data: a List of histograms with B bins
	K: number of clusters
	L: number of histograms
	B: number of bins
	thresholdConvergence: likelihood difference under which we consider that we converged
	Pinit: matrix dimension K x B. Gives the histogram distribution of each cluster.
	PiInit: vector of dimension K x 1. Gives the clustering distribution.
	maxIterations: above this max, we consider that the EM algo did not converge 
	'''  
	# data needs to be of type list
	if type(data) is not list:
		data = data.tolist()     
	

	#inspired by the EM algo as coded in R package mixtools    
	numberOfRestarts = 0 #restarts if the log like does not increase
	mustRestart = False
	
	while numberOfRestarts < 50:
		
		iteration = 0
		difference = 1 + thresholdConvergence
		print( PiInit)
		P, Pi = Pinit, PiInit 
		print("Current pi parameter: ", Pi)       
		
		logScore = logLikelihood(K,L,B, P, Pi, data)
		logScores = [logScore]
		print("Initial logScore: ",logScore)
	
		while (iteration < maxIterations) and (difference > thresholdConvergence):
			
			iteration += 1
			
			#eStep
			R = expectationStep(data,K,L,B,P,Pi)
	 
			if B == 1:
				newPi = Pi

			#mStep
			P,Pi = maximizationStep(R,data,K,L,B)
			
			if B == 1:
				Pi = newPi
		
			# compute loglike score
			newLogScore = logLikelihood(K, L, B, P, Pi, data)
			difference = newLogScore - logScore
			logScores.append(newLogScore)
			
			print ("Itération number: ", iteration, "/ Last logScore: ",logScore,"/ New log score: ", newLogScore, "/ Difference attained: ", difference)
			print("Current pi parameter: ", Pi) 
			logScore = newLogScore
			
			if difference < 0:
				mustRestart = True
				print("WARNING! The log-likelihood is decrasing! ")
				break
		
		if mustRestart == True:
			print("Restarting EM...")
			numberOfRestarts +=1
			mustRestart = False
			Pinit, PiInit = initializePandPi(K,B,data,epsilonForInitialization )
			
		else:
			if iteration == maxIterations:
				print ("WARNING! The EM did not converge! ")
			else:
				print("SUCCESS! EM converged! ")
			return P,Pi,R,logScore,logScores
	return None
	print("Too many restarts of EM")


def readjustEM(data, L, B, PfromEM, PifromEM,RfromEM,  logScoreFromEM,logScoresFromEM, thresholdConvergence, epsilonForInitialization, maxIterations = 200):
	K = len(PifromEM)
	lowestIndexInPi = np.argmin(PifromEM)
	print("Relative threshold under which the cluster is considered to disappear: ", float(1/float(100*K)))
	disappearingClusters = np.where(PifromEM <float(1/float(100*K)))[0]
	print("Disappearing indexes : ", disappearingClusters)

	if K < 2 or len(disappearingClusters) == 0: 
		P = PfromEM
		Pi = PifromEM
		logScore = logScoreFromEM
		logScores = logScoresFromEM
		R= RfromEM
	
	else:
		while len(disappearingClusters)>0 and K >=2 :
			print("There exist disapearing mixture proportions! Running adjusted EM...")
			PifromEM = np.delete(PifromEM, disappearingClusters)
			PifromEM = PifromEM / float(sum(PifromEM))
			PfromEM = np.delete(PfromEM , disappearingClusters, 0)
			print("New adjusted number of clusters : ", len(PfromEM))
			K = len(PfromEM)
			P,Pi,R,logScore,logScores = expectationMaximisation(data, K, L, B, PfromEM, PifromEM, thresholdConvergence, epsilonForInitialization,  200)
			disappearingClusters = np.where(Pi <float(1/float(100*K)))[0]
			print("Disappearing indexes : ", disappearingClusters)



	return K, P, Pi,R,logScore,logScores


def readjustEMDescending(data, L, B, PfromEM, PifromEM,RfromEM,  logScoreFromEM,logScoresFromEM, thresholdConvergence, epsilonForInitialization, maxIterations = 200):
	K = len(PifromEM)
	lowestIndexInPi = np.argmin(PifromEM)
	print("Relative threshold under which the cluster is considered to disappear: ", float(1/float(100*K)))
	disappearingClusters = np.where(PifromEM <float(1/float(100*K)))[0]
	print("Disappearing indexes : ", disappearingClusters)

	if K < 2 or len(disappearingClusters) == 0: 
		P = PfromEM
		Pi = PifromEM
		logScore = logScoreFromEM
		logScores = logScoresFromEM
		R= RfromEM
	
	else:
		while len(disappearingClusters)>0 and K >=2 :
			print("There exist disapearing mixture proportions! Running adjusted EM...")
			PifromEM = np.delete(PifromEM, lowestIndexInPi)
			PfromEM = np.delete(PfromEM , lowestIndexInPi, 0)
			print("New adjusted number of clusters : ", len(PfromEM))
			K = len(PfromEM)
			P,Pi,R,logScore,logScores = expectationMaximisation(data, K, L, B, PfromEM, PifromEM, thresholdConvergence, epsilonForInitialization,  200)
			disappearingClusters = np.where(Pi <float(1/float(100*K)))[0]
			print("Disappearing indexes : ", disappearingClusters)


	return K, P, Pi,R,logScore,logScores







def shortRunsEM(data, K, L, B, thresholdConvergence, epsilonForInitialization,  maxShortRunIterations = 15, numberOfRuns = 10):
	# data needs to be of type list
	if type(data) is not list:
		data = data.tolist() 


	shortRunsPList = []
	shortRunsPiList = []
	shortRunLastLogScoreList = []
	shortRunLogScoresList = []
	shortRunDifferenceList = []

	for i in range(numberOfRuns):
		print(str(i+1), "-th short run of EM")

		Pinit, PiInit = initializePandPi(K,B,data, epsilonForInitialization)
		numberOfRestarts = 0 #restarts if the log like does not increase
		mustRestart = False
	
		while numberOfRestarts < 50:
			
			iteration = 0
			difference = 1 + thresholdConvergence
			
			P, Pi = Pinit, PiInit
			#print("Current pi parameter: ", Pi)       
		
			logScore = logLikelihood(K, L, B, P, Pi, data)
			logScores = [logScore]
			#print("P parameter: ", P)
			#print("Pi parameter: ", Pi)
			print("Initial logScore: ",logScore)

			while (iteration < maxShortRunIterations) and difference > thresholdConvergence:
			
				iteration += 1
			
				#eStep
				R = expectationStep(data,K,L,B,P,Pi)
	 
				if B == 1:
					newPi = Pi

				#mStep
				P,Pi = maximizationStep(R,data,K,L,B)
			
				if B == 1:
					Pi = newPi
		
				# compute loglike score
				newLogScore = logLikelihood(K, L, B, P, Pi, data)
				difference = newLogScore - logScore
				logScores.append(newLogScore)
			
				#print ("Itération number: ", iteration, "/ Last logScore: ",logScore,"/ New log score: ", newLogScore, "/ Difference attained: ", difference)
				#print("Current pi parameter: ", Pi) 
				logScore = newLogScore
				#print (iteration < maxShortRunIterations)
				#print (difference > thresholdConvergence)
			
				if difference < 0:
					mustRestart = True
					print("WARNING! The log-likelihood is decreasing! ")
					break
		
			if mustRestart == True:
				print("Restarting EM...")
				numberOfRestarts +=1
				mustRestart = False
				Pinit, PiInit = initializePandPi(K, B, data, epsilonForInitialization)
			
			else:
				shortRunsPList.append(P)
				shortRunsPiList.append(Pi)
				shortRunLastLogScoreList.append(logScore)
				shortRunLogScoresList.append(logScores)
				shortRunDifferenceList.append(difference)
				break
			print("Number of short runs restarts for EM: ", str(numberOfRestarts))


	iteration = maxShortRunIterations
	#print(shortRunsPList, "  --  ", shortRunsPiList, "  --  ", shortRunDifferenceList,  "  --  ",shortRunLastLogScoreList, "  --  ", shortRunLogScoresList, "  --  ", iteration)
	return shortRunsPList, shortRunsPiList, shortRunDifferenceList, shortRunLastLogScoreList, shortRunLogScoresList, iteration



def bestShortRunsEMParameters(shortRunsPList, shortRunsPiList, shortRunDifferenceList, shortRunLastLogScoreList, shortRunLogScoresList, iteration):
	index = np.argmax(np.array(shortRunLastLogScoreList))
	P = shortRunsPList[index]
	Pi = shortRunsPiList[index]
	difference = shortRunDifferenceList[index]
	logScore = shortRunLastLogScoreList[index]
	logScores = shortRunLogScoresList[index]
	return P, Pi, difference, logScore, logScores


	
def expectationMaximisation2(data, K, L, B, thresholdConvergence, epsilonForInitialization,  maxIterations = 100):
	'''
	Enter some data of dimension L x B. Apply the EM algorithm in order to get the parameters of the multinomial mixture.

	data: a List of histograms with B bins
	K: number of clusters
	L: number of histograms
	B: number of bins
	thresholdConvergence: likelihood difference under which we consider that we converged
	Pinit: matrix dimension K x B. Gives the histogram distribution of each cluster.
	PiInit: vector of dimension K x 1. Gives the clustering distribution.
	maxIterations: above this max, we consider that the EM algo did not converge 
	'''     
	
	# data needs to be of type list
	if type(data) is not list:
		data = data.tolist()     
	

	#inspired by the EM algo as coded in R package mixtools    
	numberOfRestarts = 0 #restarts if the log like does not increase
	mustRestart = False
	
	if K <=50:
		shortRunsPList, shortRunsPiList, shortRunDifferenceList, shortRunLastLogScoreList, shortRunLogScoresList, iteration = shortRunsEM(data, K, L, B, epsilonForInitialization, thresholdConvergence)
	else:
		shortRunsPList, shortRunsPiList, shortRunDifferenceList, shortRunLastLogScoreList, shortRunLogScoresList, iteration = shortRunsEM(data, K, L, B, epsilonForInitialization, thresholdConvergence, 15, 25)
	P, Pi, difference, logScore, logScores= bestShortRunsEMParameters(shortRunsPList, shortRunsPiList, shortRunDifferenceList, shortRunLastLogScoreList, shortRunLogScoresList, iteration)
	R = expectationStep(data,K,L,B,P,Pi)
   
	while numberOfRestarts < 50:   
		logScore = logLikelihood(K,L,B, P, Pi, data)
		logScores.append(logScore)

		#print "Enters in the main while loop ? " + str((iteration < maxIterations) and (difference > thresholdConvergence))
		while (iteration < maxIterations) and (difference > thresholdConvergence):
			
			iteration += 1
			
			
	 
			#mStep
			P,Pi = maximizationStep(R,data,K,L,B)

            
		
			# compute loglike score
			newLogScore = logLikelihood(K, L, B, P, Pi, data)
			difference = newLogScore - logScore
			logScores.append(newLogScore)
			
			print ("Itération number: ", iteration, "/ Last logScore: ",logScore,"/ New log score: ", newLogScore, "/ Difference attained: ", difference)
			print("Current pi parameter: ", Pi) 
			logScore = newLogScore
			
			if difference < 0:
				mustRestart = True
				print("WARNING! The log-likelihood is decrasing! ")
				break
                
            #eStep
			R = expectationStep(data,K,L,B,P,Pi)
		
		if mustRestart == True:
			print("Restarting EM...")
			numberOfRestarts +=1
			mustRestart = False
			P, Pi = initializePandPi(K,B,data,epsilonForInitialization)
			difference = 1 + thresholdConvergence
			
		else:
			if iteration == maxIterations:
				print ("WARNING! The EM did not converge! ")
			else:
				print("SUCCESS! EM converged! ")
			return P,Pi,R,logScore,logScores
	return None
	print("Too many restarts of EM")



