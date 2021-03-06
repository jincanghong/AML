import networkx as nx
import numpy as np
import random

def getTransitionMatrix(network, nodes):
	matrix = np.empty([len(nodes), len(nodes)])

	for i in range(0, len(nodes)):
		neighs = list(network.neighbors(nodes[i]))
		sums = 0
		for neigh in neighs:
			sums += network[nodes[i]][neigh]['weight']

		for j in range(0, len(nodes)):
			if i == j :
				matrix[i,j] = 0
			else:
				if nodes[j] not in neighs:
					matrix[i, j] = 0
				else:
					matrix[i, j] = network[nodes[i]][nodes[j]]['weight'] / sums

	return matrix

def generateSequence(startIndex, transitionMatrix, path_length, alpha, israndom=0):
	result = [startIndex]
	current = startIndex

	for i in range(0, path_length):
		if random.random() < alpha:
			nextIndex = startIndex
		else:
			probs = transitionMatrix[current]
			probs /= sum(probs)
			if israndom == 0:  # not random walk
				nextIndex = np.random.choice(len(probs), 1, p=probs)[0]
			else:  # random walk 
				nextIndex = np.random.choice(len(probs), 1, p=[1/len(probs)]*len(probs))[0] 
    
		result.append(nextIndex)
		current = nextIndex

	return result

def random_walk(G, num_paths, path_length, alpha, israndom):
	nodes = list(G.nodes())
	transitionMatrix = getTransitionMatrix(G, nodes)

	sentenceList = []

	for i in range(0, len(nodes)):
		for j in range(0, num_paths):
			indexList = generateSequence(i, transitionMatrix, path_length, alpha, israndom)
			sentence = [int(nodes[tmp]) for tmp in indexList]
			sentenceList.append(sentence)

	return sentenceList