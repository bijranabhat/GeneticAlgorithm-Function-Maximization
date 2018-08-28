import numpy as np; 
import pandas as pd
import random
import os
import time
from math import inf
import operator

#Add own working directory
os.chdir("Tests")

class GeneticAlg():

	def __init__(self, generationSize = 10, numGenerations = 100, timeoutMin = 5):
		self.GenerationSize = generationSize
		self.NumGenerations = numGenerations
		self.TimeoutMin = timeoutMin
		self.readData()
		self.Fitness = -inf
		print("\nMUTATION \n")

		#Set random seed
		#np.random.seed(random.randint(0,100))
		self.evolve(generationSize, numGenerations)

		#self.printCoefficients()

	#Reads data from a file
	def readData(self):
		print("Welcome to the genetic algorithm builder.\n" +
			  "Please enter a filename for a function you would like to solve.\n" +
			  "The file should start with the number (n) of lines, followed by a space\n" +
			  "and then a list of numbers to fill the array")

		while(True):

			try:
				#Get filename
				filename = input().strip()

				#Read in all lines of data
				out = open(filename, "r")
				file =  out.readlines()
				out.close()

				#Filter out all empty lines
				file = [line for line in file if line.strip()]

				#Get number of variables
				self.NumVars = int(file[0].strip())

				#Initialize best variables
				self.BestVars = np.concatenate([np.array([1]),np.random.uniform(0,10,self.NumVars)])

				#initialize final Array
				self.Array = []

				#Show the array read in
				print("\nARRAY READ SUCCESSFULLY. INFORMATION BELOW.\n")
				print("Generation Size:", self.GenerationSize)
				print("Number of Generations:", self.NumGenerations)
				print("Number of Variables:",self.NumVars)
				print("Time Limit:",self.TimeoutMin)

				print("Array:")

				#Read in each line of data for array
				for i in range(1,self.NumVars + 2):

					#Formate data correctly
					row = str(file[i]).rstrip("\n")
					row = [int(num) for num in row.strip().split(" ") if num.strip()]

					#Verify row is formatted correctly
					print(row)
					self.Array.append(row)

				#Return array
				return

			except Exception as e:
				print("\nError! file read failed. See details and try another filename.\n")
				print(str(e) + "\n")



		self.BestVars = np.concatenate([np.array([1]),np.random.uniform(0,10,self.NumVars)]).astype(int)

	#Print coefficients to verify readData is working correctly
	def printCoefficients(self):
		for i in range(len(self.Array)):
			for j in range(i,len(self.Array)):
				print(self.Array[i][j], sep = " ")

	#Determine the fitness of the specific weights
	def fitness(self, Coeffs):
		fitness = 0
		for i in range(len(self.Array)):
			for j in range(i,len(self.Array)):
				if j == 0:
					fitness = self.Array[i][j]
				else:
					fitness += self.Array[i][j] * Coeffs[i] * Coeffs[j]

		return fitness

	#Pads binary string so that it is always 8 characters
	def padBinString(self,num):
		temp_str = str(bin(num))[2 :] 							# String of number
		final_string = '0' * (8 - len(temp_str)) + temp_str 	# zero-padding plus string number

		return final_string

	#Converts an array of values to a binary string
	def convertToBinary(self,arr):

		#Initialize variables and empty final string
		variables = arr[1:]
		final_var_string = ""

		#Iterate through variables
		for var in variables:
			#Add the padded binary string to final
			final_var_string += self.padBinString(var)

		return final_var_string

	#Convert a binary string to an integer array
	def convertBinToVars(self, kid):
		#Initialize array of all ones
		arr = np.array([1]*(self.NumVars + 1))

		#For each variable
		for var in range(1,self.NumVars + 1):
			arr[var] = int(kid[(var-1)*8:(var)*8], base = 2)

		return arr.astype(int)


	#Use combination to generate a child
	def generateChild(self, dad, mom):
		#Get middle index
		mid_index = len(dad) // 2

		# Get random number indices for each half
		mut_index_first_half = np.random.randint(0, mid_index)
		mut_index_second_half = np.random.randint(mid_index, len(dad))

		#Create two children with crossover
		kid1 = dad[: mut_index_first_half] + mom[mut_index_first_half:mut_index_second_half] + dad[mut_index_second_half:]
		kid2 = mom[: mut_index_first_half] + dad[mut_index_first_half:mut_index_second_half] + mom[mut_index_second_half:]

		#Store potential kids for future random choosing
		potential_kids = [kid1,kid2]

		#choose a random child from potential kids
		return potential_kids[np.random.randint(0,2)]

	#Mutate randomly in a population
	def mutate(self,spawn,oddsOfMutation):

		#For each child in the generation
		for offspring in range(self.GenerationSize):
			#Randomly choose a child
			randKid = np.random.randint(0,self.GenerationSize)

			#Potentially mutate that child based on the odds of mutation ()
			spawn[randKid] = self.mutateOne(spawn[randKid],oddsOfMutation)

		return spawn


	#Mutate random children
	def mutateOne(self, kid, oddsOfMutation):
		# randomly select how many times to mutate
		# during each time, randomly select a bit to mutate
		def subMutateOneV1(kid):
			#Random number of mutation changes
			mutNum = np.random.randint(len(kid))

			#Mutate the child "changeNum" times
			for change in range(mutNum):
				#Random index
				randIndex = np.random.randint(len(kid))

				if kid[randIndex] == "1":
					kid = kid[:randIndex] + "0" + kid[randIndex + 1 :]
				else:
					kid = kid[:randIndex] + "1" + kid[randIndex + 1:]

			return kid

		# randomly select a pair of indices
		# flip all bits between those two indices
		def subMutateOneV2(kid):
			index1 = np.random.randint(len(kid))
			index2 = np.random.randint(len(kid))

			index1, index2 = min(index1, index2), max(index1, index2)

			for index in range(index1, index2):
				if kid[index] == '0':
					kid = kid[: index] + '1' + kid[index + 1 :]
				else:
					kid = kid[: index] + '0' + kid[index + 1 :]

			return kid

		# randomly select a pair of indices
		# randomly decide whether to flip each bit
		# between those two indices
		def subMutateOneV3(kid):
			index1 = np.random.randint(len(kid))
			index2 = np.random.randint(len(kid))

			index1, index2 = min(index1, index2), max(index1, index2)

			for index in range(index1, index2):
				flip = np.random.randint(2)
				if flip == 0:
					if kid[index] == '0':
						kid = kid[: index] + '1' + kid[index + 1 :]
					else:
						kid = kid[: index] + '0' + kid[index + 1 :]

			return kid

		methods = [subMutateOneV1, subMutateOneV2, subMutateOneV3]

		#Turn kid into binary
		kid = self.convertToBinary(kid)

		randomChance = np.random.randint(0,oddsOfMutation)

		#Only do it if the random Chance equals 0
		if randomChance == 0:

			#kid = subMutateOneV1(kid)
			#kid = subMutateOneV2(kid)
			#kid = subMutateOneV3(kid)

			methodID = np.random.randint(len(methods))
			kid = methods[methodID](kid)

		#Put child back in population
		return self.convertBinToVars(kid)

	def createOffspring(self,spawn, fitness, generationSize):
		newspawn = []
		for kid in range(generationSize):
			dad, mom = self.getParents(spawn, fitness)
			newspawn.append(self.convertBinToVars(self.generateChild(dad,mom)))

		return newspawn

	def chooseParentIndex(self,fitChart):
		rNum = np.random.uniform(0,1)

		for i in range(len(fitChart)):
			if fitChart[i] >= rNum:
				return (i - 1)

		#If it was exactly one
		return len(fitChart) - 1

	#Cumulate (sum) distribution of fitnesses
	def getFitChart(self,fitnessPerc):
		fitChart = [fitnessPerc[0]]
		for i in range(1,len(fitnessPerc)):
			fitChart.append(sum(fitnessPerc[0:i+1]))

		return fitChart

	def putInBounds(self,num):
		if num < 0:
			return 0
		elif num > 255:
			return 255

		return num

	#Fix inbreeding if the spawn grow stale
	def fixInbreeding(self):
		new_spawn = []

		for kid in range(self.GenerationSize):
			new_kid = np.array([1]*(self.NumVars + 1))
			for i in range(1, (self.NumVars+1)):
				new_kid[i] = self.putInBounds(np.random.normal(self.BestVars[i],1))
			new_spawn.append(new_kid)


		return new_spawn

	# Find the parents from a population probabilistically

	def getParents(self, spawn, fitness):

		#Get fitness percentages (all non negative)
		pos_fitness = [(fit + (abs(min(fitness)) + 1)) for fit in fitness]
		fitness_perc = (pos_fitness / sum(pos_fitness))

		#Get the fitness chart
		fit_chart = self.getFitChart(fitness_perc)
		#print(" ".join([str(i) for i in fitChart]))

		while(True):
			dadIndex, momIndex = self.chooseParentIndex(fit_chart), self.chooseParentIndex(fit_chart)
			if dadIndex != momIndex:
				return self.convertToBinary(spawn[dadIndex]), self.convertToBinary(spawn[momIndex])


	def evolve(self,generationSize, numGenerations):

		#Start time for execution
		startTime = time.time()

		#How often mutations should occur
		mutateOccurence = 10

		#Count of how many generations had no change
		noChange = 0

		#Initialize container for spawn
		spawn = []

		#Create initial random offspring
		for offspring in range(generationSize):
			spawn.append(np.concatenate([np.array([1]),np.random.randint(0,256, size = self.NumVars)]))

		#Generate the fitness of each child
		fitness = [self.fitness(coeffs) for coeffs in spawn]

		#Pick parents randomly based on fitness
		dad, mom = self.getParents(spawn, fitness)

		#Print header information for output
		print('{:<5}  {:<10} {:<15}  {:<98}'.format("Gen", "Timer", "Fitness", "Variables"))

		#Evolve the sample
		for generation in range(numGenerations):
			hyper_spawn = []

			#Generate children
			#print(spawn)
			spawn = self.createOffspring(spawn,fitness,generationSize)
			spawn = self.mutate(spawn,mutateOccurence)

			# print(self.convertBinToVars(dad), self.convertBinToVars(mom))
			# print(" ".join([str(i) for i in spawn]))
			# print("\n\n\n\n\n")

		 	#Generate the fitness of each child
			fitness = [self.fitness(kid) for kid in spawn]

			#Update max fitness if better option is found
			if (max(fitness) > self.Fitness):

				#Update Fitness and variable values
				self.GensToBest = generation
				self.Fitness = max(fitness)
				self.BestVars = spawn[fitness.index(self.Fitness)]

			#Update spawn if population grows stale after awhile
			else:
				noChange += 1
				if noChange == 5:
					noChange = 0
					spawn = self.fixInbreeding()


			#Add best candidate to hyper_spawn
			hyper_spawn.append(spawn[fitness.index(max(fitness))])

			#Check to see if hyperspawn should replace spawn
			if len(hyper_spawn) == self.GenerationSize:
				spawn = hyperspawn

			#Calculate time left
			time_left = (self.TimeoutMin * 60) - int(time.time() - startTime)

			#Print out the fitness
			outputInfo = '{:<5} {:<10} {:<16}  {:<100}'.format(generation,
														str((self.TimeoutMin * 60) - int(time.time() - startTime)),
														"{0:.2f}".format(round(self.Fitness,2)),
														str(self.BestVars))
			print(outputInfo)

			#If the timer has run out, end the cycle
			if (time_left < 0):
				print("\nTIMEOUT REACHED BEFORE GENERATIONS FINISHED. EXITING.")
				break


		#Print output
		print("\nEVOLUTION FINISHED.\n\n" +
			  "Generations Until Optimum Found: " + str(self.GensToBest) +
			  "\nMaximum Fitness: " + str(self.Fitness) +
			  "\nBest Variables: " + str(self.BestVars))

		return

GeneticAlg(generationSize = 30, numGenerations = 20000, timeoutMin = 0.1)