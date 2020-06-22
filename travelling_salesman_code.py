# File:  solve_tsp_3.py
#
# This script will solve a TSP via 3 different methods:  nearest neighbor (NN) heuristic, IP, or simulated annealing (SA)
#
# Inputs:
# 	locationsFolder:	For example, practice_3
#	objectiveType:		1 --> Minimize Distance, 2 --> Minimize Time
#	solveNN				1 --> Solve using NN
#	solveIP				1 --> Solve using IP
#	solveSA				1 --> Solve using SA
# 	IPcutoffTime:		-1 --> No time limit, o.w., max number of seconds for Gurobi to run
# 	turnByTurn:			1 --> Use MapQuest for detailed routes. 0 --> Just draw straight lines between nodes.
#
# How to run:
# 	python solve_tsp_3.py practice_3 1 1 1 1 120 1
# python solve_tsp_3.py practice_25 1 0 0 1 120 1

import sys			# Allows us to capture command line arguments
import csv
import folium		# https://github.com/python-visualization/folium

import urllib2
import json
import pandas as pd
from pandas.io.json import json_normalize

from collections import defaultdict

from gurobipy import *
from random import randint
from random import random
import math
import time
import matplotlib.pyplot as plt
from numpy.random import randn


# -----------------------------------------
mapquestKey = 'bUUOqGZ0MdRTAFXPGWKgcPKNsrUQGzcO'			# Visit https://developer.mapquest.com/ to get a free API key
#print "YOU MUST SET YOUR MAPQUEST KEY"
#exit()


# Put your SA parameters here:
#	Tzero:				Initial temperature for SA
#	I:					Number of iterations per temperature for SA
#	delta:				Cooling schedule temp reduction for SA
#	Tfinal:				Minimum allowable temp for SA
#	SAcutoffTime:		Number of seconds to allow your SA heuristic to run
Tzero = 500
I = 50000
delta = 0.01
Tfinal = 0
SAcutoffTime = 120
# -----------------------------------------

# http://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries-in-python
def make_dict():
	return defaultdict(make_dict)

class make_node:
	def __init__(self, nodeName, isDepot, latDeg, lonDeg, demand):
		# Set node[nodeID]
		self.nodeName 	= nodeName
		self.isDepot	= isDepot
		self.latDeg		= latDeg
		self.lonDeg		= lonDeg
		self.demand		= demand

def genTravelMatrix(coordList, locIDlist, all2allStr, one2manyStr, many2oneStr):
	# We'll use MapQuest to calculate.
	transportMode = 'fastest'
	# Other option includes:  'pedestrian' (apparently 'bicycle' has been removed from the API)
	routeTypeStr = 'routeType:%s' % transportMode

	# Assemble query URL
	myUrl = 'http://www.mapquestapi.com/directions/v2/routematrix?'
	myUrl += 'key={}'.format(mapquestKey)
	myUrl += '&inFormat=json&json={locations:['

	# Insert coordinates into the query:
	n = len(coordList)
	for i in range(0,n):
		if i != n-1:
			myUrl += '{{latLng:{{lat:{},lng:{}}}}},'.format(coordList[i][0], coordList[i][1])
		elif i == n-1:
			myUrl += '{{latLng:{{lat:{},lng:{}}}}}'.format(coordList[i][0], coordList[i][1])
	myUrl += '],options:{{{},{},{},{},doReverseGeocode:false}}}}'.format(routeTypeStr, all2allStr,one2manyStr,many2oneStr)

	# print "\nThis is the URL we created.  You can copy/paste it into a Web browser to see the result:"
	# print myUrl


	# Now, we'll let Python go to mapquest and request the distance matrix data:
	request = urllib2.Request(myUrl)
	response = urllib2.urlopen(request)
	data = json.loads(response.read())

	# print "\nHere's what MapQuest is giving us:"
	# print data

	# This info is hard to read.  Let's store it in a pandas dataframe.
	# We're goint to create one dataframe containing distance information:
	distance_df = json_normalize(data, "distance")
	# print "\nHere's our 'distance' dataframe:"
	# print distance_df

	# print "\nHere's the distance between the first and second locations:"
	# print distance_df.iat[0,1]

	# Our dataframe is a nice table, but we'd like the row names (indexes)and column names to match our location IDs.
	# This would be important if our locationIDs are [1, 2, 3, ...] instead of [0, 1, 2, 3, ...]
	distance_df.index = locIDlist
	distance_df.columns = locIDlist

	# Now, we can find the distance between location IDs 1 and 2 as:
	# print "\nHere's the distance between locationID 1 and locationID 2:"
	# print distance_df.loc[1,2]


	# We can create another dataframe containing the "time" information:
	time_df = json_normalize(data, "time")

	# print "\nHere's our 'time' dataframe:"
	# print time_df

	# Use our locationIDs as row/column names:
	time_df.index = locIDlist
	time_df.columns = locIDlist


	# We could also create a dataframe for the "locations" information (although we don't need this for our problem):
	#print "\nFinally, here's a dataframe for 'locations':"
	#df3 = json_normalize(data, "locations")
	#print df3

	return(distance_df, time_df)


def genShapepoints(startCoords, endCoords):
	# We'll use MapQuest to calculate.
	transportMode = 'fastest'		# Other option includes:  'pedestrian' (apparently 'bicycle' has been removed from the API)

	# assemble query URL
	myUrl = 'http://www.mapquestapi.com/directions/v2/route?key={}&routeType={}&from={}&to={}'.format(mapquestKey, transportMode, startCoords, endCoords)
	myUrl += '&doReverseGeocode=false&fullShape=true'

	# print "\nThis is the URL we created.  You can copy/paste it into a Web browser to see the result:"
	# print myUrl

	# Now, we'll let Python go to mapquest and request the distance matrix data:
	request = urllib2.Request(myUrl)
	response = urllib2.urlopen(request)
	data = json.loads(response.read())

	# print "\nHere's what MapQuest is giving us:"
	# print data

	# retrieve info for each leg: start location, length, and time duration
	lats = [data['route']['legs'][0]['maneuvers'][i]['startPoint']['lat'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	lngs = [data['route']['legs'][0]['maneuvers'][i]['startPoint']['lng'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	secs = [data['route']['legs'][0]['maneuvers'][i]['time'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	dist = [data['route']['legs'][0]['maneuvers'][i]['distance'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]

	# print "\nHere are all of the lat coordinates:"
	# print lats

	# create list of dictionaries (one dictionary per leg) with the following keys: "waypoint", "time", "distance"
	legs = [dict(waypoint = (lats[i],lngs[i]), time = secs[i], distance = dist[i]) for i in range(0,len(lats))]

	# create list of waypoints (waypoints define legs)
	wayPoints = [legs[i]['waypoint'] for i in range(0,len(legs))]
	# print wayPoints

	# get shape points (each leg has multiple shapepoints)
	shapePts = [tuple(data['route']['shape']['shapePoints'][i:i+2]) for i in range(0,len(data['route']['shape']['shapePoints']),2)]
	# print shapePts

	return shapePts

def solve_tsp_nn(objectiveType, distance_df, time_df, node):
	# We're going to find a route and a cost for this route
	#Change the dataframe into dictionary
	distanceDict=distance_df.to_dict()
	timeDict=time_df.to_dict()
	nn_route = []
	nn_value=-1
	#Select the cost type
	if objectiveType==1:
		inputDict=distanceDict
	elif objectiveType==2:
		inputDict=timeDict
	my_tour=[]
	depot=-1
	#Find the depot
	for i in range(0,len(node)):
		if node[i].isDepot==1:
			my_tour.append(i)
			depot=i
	to_visit=range(len(inputDict))
	point=depot
	total_distance=0
	#take a loop unless we reach the last place to visit
	while(len(to_visit)>1):
		distance=[]
		pos=[]
		min_distance=1000000
		min_pos=-1
		#find the list of distance we can go
		for j in range(0,len(inputDict)):
			#print "j: ",j
			if j!= point and j not in my_tour:
				distance.append(inputDict[j][point])
				pos.append(j)
		#find the minimum of those distance
		for k in range(0,len(distance)):
			if distance[k]<min_distance:
				min_distance=distance[k]
				min_pos=pos[k]
		total_distance+=min_distance
		my_tour.append(min_pos)
		to_visit.remove(min_pos)
		point=min_pos
	#Make the final tour
	my_tour.append(depot)
	total_distance+=inputDict[depot][point]
	nn_route=my_tour
	nn_cost=total_distance

	return (nn_route, nn_cost)

def solve_tsp_ip(objectiveType, distance_df, time_df, node, cutoffTime):
	ip_route = []

	N = []
	q = 0
	for nodeID in node:
		N.append(nodeID)
		if (node[nodeID].isDepot == 0):
			q += 1

	c = defaultdict(make_dict)
	decvarx = defaultdict(make_dict)
	decvaru = {}


	# GUROBI
	m = Model("TSP")

	# The objective is to minimize the total travel distance.
	m.modelSense = GRB.MINIMIZE

	# Give Gurobi a time limit:
	if (cutoffTime > 0):
		m.params.TimeLimit = cutoffTime

	# Define our decision variables (with their objective function coefficients:
	for i in N:
		decvaru[i] = m.addVar(lb=1, ub=q+2, obj=0, vtype=GRB.CONTINUOUS, name="u.%d" % (i))
		for j in N:
			if (i != j):
				if (objectiveType == 1):
					# We want to minimize distance
					decvarx[i][j] = m.addVar(lb=0, ub=1, obj=distance_df.loc[i,j], vtype=GRB.BINARY, name="x.%d.%d" % (i,j))
				else:
					# We want to minimize time
					decvarx[i][j] = m.addVar(lb=0, ub=1, obj=time_df.loc[i,j], vtype=GRB.BINARY, name="x.%d.%d" % (i,j))

	# Update model to integrate new variables:
	m.update()

	# Build our constraints:
	# Constraint (2)
	for i in N:
		m.addConstr(quicksum(decvarx[i][j] for j in N if j != i) == 1, "Constr.2.%d" % i)

	# Constraint (3)
	for j in N:
		m.addConstr(quicksum(decvarx[i][j] for i in N if i != j) == 1, "Constr.3.%d" % j)

	# Constraint (4)
	for i in range(1, q+1):
		for j in N:
			if (j != i):
				m.addConstr(decvaru[i] - decvaru[j] + 1 <= (q + 1)*(1 - decvarx[i][j]), "Constr.4.%d.%d" % (i,j))

	# Solve
	m.optimize()


	if (m.Status == GRB.INFEASIBLE):
		# NO FEASIBLE SOLUTION EXISTS

		print "Sorry, Guroby is telling us this problem is infeasible."

		ip_cost = -999	# Infeasible

	elif ((m.Status == GRB.TIME_LIMIT) and (m.objVal > 1e30)):
		# NO FEASIBLE SOLUTION WAS FOUND (maybe one exists, but we ran out of time)

		print "Guroby can't find a feasible solution.  It's possible but one exists."

		ip_cost = -888	# Possibly feasible, but no feasible solution found.

	else:
		# We found a feasible solution
		if (m.objVal == m.ObjBound):
			print "Hooray...we found an optimal solution."
			print "\tOur objective function value:  %f" % (m.objVal)
		else:
			print "Good News:  We found a feasible solution."
			print "Bad News:  It's not provably optimal."
			print "\tOur objective function value:  %f" % (m.objVal)
			print "\tGurobi's best bound: %f" % (m.ObjBound)

		if (m.Status == GRB.TIME_LIMIT):
			print "\tGurobi reached it's time limit"


		# Let's use the values of the x decision variables to create a tour:
		startNode = -1
		for nodeID in node:
			if (node[nodeID].isDepot):
				startNode = nodeID

		if (startNode == -1):
			# We didn't find a depot in our list of locations.
			# We'll let our starting node be the first node
			startNode = min(node)
			print "I couldn't find a depot.  Starting the tour at locationID %d" % (startNode)

		allDone = False
		i = startNode
		ip_route.append(i)
		while (not allDone):
			for j in N:
				if (i != j):
					if (decvarx[i][j].x > 0.9):
						# We traveled from i to j
						ip_route.append(j)
						i = j
						break	# We found the link from i to j.  Stop the "for" loop
			if (len(ip_route) == len(N)):
				# We found all the customers
				allDone = True

		# Now, add a link back to the start
		ip_route.append(startNode)

		ip_cost = m.objVal
	return (ip_route, ip_cost)

#Simulated Annealing Function
def solve_tsp_sa(objectiveType, distance_df, time_df, node, Tzero, I, delta, Tfinal, SAcutoffTime):
	#Initialise variable and list including start time
	start_time=time.time()
	sa_route = []
	sa_cost = -999
	cur_list=[]
	best_list=[]
	point_list=[]
	i_good_list=[]
	i_bad_list=[]
	i_worst_list=[]
	check_list=[]
	i_good_point=[]
	i_bad_point=[]
	i_worst_point=[]
	i_list=[]
	#Call the function of nearest neighbour to get the nearest neighbour tour
	[nn_route, nn_cost] = solve_tsp_nn(objectiveType, distance_df, time_df, node)
	#Check the depot and assign it as start node
	startNode = -1
	for nodeID in node:
		if (node[nodeID].isDepot):
			startNode = nodeID
	#Initialise the varibales used in the algorithm
	Xzero=nn_route
	Xcur=Xzero
	Zcur=nn_cost
	Tcur=Tzero
	Xbest=Xcur
	Zbest=Zcur
	#Start the iteration
	for count in range(1,I+1):
		i_list.append(count)
		good_flag=False
		bad_flag= False
		worst_flag = False
		c = 0
		Zcount=10000000
		#Generate subtours
		while (c<5):
			#Generate random start and end point as a and b
			a = randint(1,len(Xcur)-3)
			b = randint(a+1,len(Xcur)-2)
			output_list=[]
			cost=0
			start_list=Xcur[:a]
			#the part to be reversed
			rev_list= Xcur[a:b+1]
			end_list = Xcur[b+1:]
			#reverse the pecific list
			add_list= rev_list[::-1]
			#Generate subtour
			output_list.extend(start_list)
			output_list.extend(add_list)
			output_list.extend(end_list)
			#Check if the subtour is unique
			if(output_list not in check_list):
				c+=1
				check_list.append(output_list)
				#Select the ditance or time matrix and calculate its cost
				if objectiveType==1:
					for i in range(0,len(output_list)-1):
						cost+= distance_df[output_list[i]][output_list[i+1]]
				else:
					for i in range(0,len(output_list)-1):
						cost+= time_df[output_list[i]][output_list[i+1]]
				#Accpet the best subtour fromt he loop
				if(cost<Zcount):
					Xcount=output_list
					Zcount=cost
		check_list=[]
		#Check if the value from the subtour is better than current solution
		if(Zcount<Zcur):
			Xcur = Xcount
			Zcur = Zcount
			good_flag=True
			i_good_list.append(count)
			i_good_point.append(Zcount)
		else:
			#If the subtour is not better check if we can still accept it
			delC = Zcount-Zcur
			if(random()<=math.exp(-delC/Tcur)):
				Xcur = Xcount
				Zcur = Zcount
				bad_flag=True
				i_bad_list.append(count)
				i_bad_point.append(Zcount)
			else:
				worse_flag=True
				i_worst_list.append(count)
				i_worst_point.append(Zcount)
		#If the current tour is better than the best, replace it.
		if(Zcur < Zbest):
			Zbest = Zcur
			Xbest = Xcur
		#Flags to check which solution we got for plotting
		if(good_flag==False):
			i_good_list.append(count)
			i_good_point.append(None)
		if(bad_flag==False):
			i_bad_list.append(count)
			i_bad_point.append(None)
		if(worst_flag== False):
			i_worst_list.append(count)
			i_worst_point.append(None)
		cur_list.append(Zcur)
		best_list.append(Zbest)
		#Decrease the temperature
		Tcur = Tcur-delta
		#Breaking conditions
		if (Tcur<Tfinal or time.time()-start_time>= SAcutoffTime):
			break
	sa_route=Xbest
	sa_cost=Zbest
	#Plot the points
	plt.plot(i_bad_list, i_bad_point, 'go', label = 'Bad Value Accepted')
	plt.plot(i_good_list, i_good_point, 'bo', label = 'Good Value Accepted')
	plt.plot(i_worst_list, i_worst_point, 'ro', label='Bad Value Rejected')
	plt.plot(i_list,cur_list, color='orange', label='Current Value')
	plt.plot(i_list, best_list, color = 'purple', label = 'Best Value')
	plt.xlabel("Iterations")
	plt.ylabel("Objective Value")
	plt.legend()
	runtime=time.time()-start_time
	plt.show()
	print sa_cost
	exit()
	print "Runtime of SA: %d seconds " %(runtime)
	return (sa_route, sa_cost)


# Capture command line inputs:
if (len(sys.argv) == 8):
	locationsFolder		= str(sys.argv[1])		# Ex:  practice_3
	objectiveType		= int(sys.argv[2])		# 1 --> Minimize Distance, 2 --> Minimize Time
	solveNN				= int(sys.argv[3])		# 1 --> Solve NN.  0 --> Don't solve NN
	solveIP				= int(sys.argv[4])		# 1 --> Solve IP.  0 --> Don't solve IP
	solveSA				= int(sys.argv[5])		# 1 --> Solve SA.  0 --> Don't solve SA
	IPcutoffTime		= float(sys.argv[6])	# -1 --> No time limit, o.w., max number of seconds for Gurobi to run
	turnByTurn			= int(sys.argv[7])		# 1 --> Use MapQuest for detailed routes.  0 --> Just draw straight lines connecting nodes.
	if (objectiveType not in [1,2]):
		print 'ERROR:  objectiveType %d is not recognized.' % (objectiveType)
		print 'Valid numeric options are:  1 (minimize distance) or 2 (minimize time)'
		quit()
else:
	print 'ERROR: You passed', len(sys.argv)-1, 'input parameters.'
	quit()


# Initialize a dictionary for storing all of our locations (nodes in the network):
node = {}


# Read location data
locationsFile = 'Problems/%s/tbl_locations.csv' % locationsFolder
# Read problem data from .csv file
# NOTE:  We are assuming that the .csv file has a pre-specified format.
#	 Column 0 -- nodeID
# 	 Column 1 -- nodeName
#	 Column 2 -- isDepot (1 --> This node is a depot, 0 --> This node is a customer
#	 Column 3 -- lat [degrees]
#	 Column 4 -- lon [degrees]
#	 Column 5 -- Customer demand
with open(locationsFile, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
		if (row[0][0] != '%'):
			nodeID = int(row[0])
			nodeName = str(row[1])
			isDepot = int(row[2])
			latDeg = float(row[3])
			lonDeg = float(row[4])
			demand = float(row[5])

			node[nodeID] = make_node(nodeName, isDepot, latDeg, lonDeg, demand)

# Use MapQuest to generate two pandas dataframes.
# One dataframe will contain a matrix of travel distances,
# the other will contain a matrix of travel times.
coordList = []
locIDlist = []
for i in node:
	coordList.append([node[i].latDeg, node[i].lonDeg])
	locIDlist.append(i)

all2allStr	= 'allToAll:true'
one2manyStr	= 'oneToMany:false'
many2oneStr	= 'manyToOne:false'

[distance_df, time_df] = genTravelMatrix(coordList, locIDlist, all2allStr, one2manyStr, many2oneStr)


# Now, solve the TSP:
[nn_route, ip_route, sa_route, nn_cost, ip_cost, sa_cost] = [[], [],[], -1, -1,-1]

if (solveNN):
	# Solve the TSP using nearest neighbor
	[nn_route, nn_cost] = solve_tsp_nn(objectiveType, distance_df, time_df, node)
if (solveIP):
	# Solve the TSP using the IP model
	[ip_route, ip_cost] = solve_tsp_ip(objectiveType, distance_df, time_df, node, IPcutoffTime)
if (solveSA):
	# Solve the TSP using simulated annealing
	[sa_route, sa_cost] = solve_tsp_sa(objectiveType, distance_df, time_df, node, Tzero, I, delta, Tfinal, SAcutoffTime)		# FIXME -- YOU'LL NEED TO PASS SOME INFO TO THIS FUNCTION




# Create a map of our solution
mapFile = 'Problems/%s/osm.html' % locationsFolder
map_osm = folium.Map(location=[node[0].latDeg, node[0].lonDeg], zoom_start=10)

# Plot markers
for nodeID in node:
	if (node[nodeID].isDepot):
		folium.Marker([node[nodeID].latDeg, node[nodeID].lonDeg], icon = folium.Icon(color ='red'), popup = node[nodeID].nodeName).add_to(map_osm)
	else:
		folium.Marker([node[nodeID].latDeg, node[nodeID].lonDeg], icon = folium.Icon(color ='blue'), popup = node[nodeID].nodeName).add_to(map_osm)

if (turnByTurn):
	# (PRETTY COOL) Plot turn-by-turn routes using MapQuest shapepoints:
	if (nn_cost > 0):
		# a) nearest neighbor:
		i = nn_route[0]
		for j in nn_route[1:]:
			startCoords = '%f,%f' % (node[i].latDeg, node[i].lonDeg)
			endCoords = '%f,%f' % (node[j].latDeg, node[j].lonDeg)

			myShapepoints = genShapepoints(startCoords, endCoords)

			folium.PolyLine(myShapepoints, color="red", weight=8.5, opacity=0.5).add_to(map_osm)

			i = j

	if (ip_cost > 0):
		# b) ip:
		i = ip_route[0]
		for j in ip_route[1:]:
			startCoords = '%f,%f' % (node[i].latDeg, node[i].lonDeg)
			endCoords = '%f,%f' % (node[j].latDeg, node[j].lonDeg)

			myShapepoints = genShapepoints(startCoords, endCoords)

			folium.PolyLine(myShapepoints, color="green", weight=4.5, opacity=0.5).add_to(map_osm)

			i = j
	if (sa_cost > 0):
		# c) simulated annealing:
		i = sa_route[0]
		for j in sa_route[1:]:
			startCoords = '%f,%f' % (node[i].latDeg, node[i].lonDeg)
			endCoords = '%f,%f' % (node[j].latDeg, node[j].lonDeg)

			myShapepoints = genShapepoints(startCoords, endCoords)

			folium.PolyLine(myShapepoints, color="blue", weight=6.5, opacity=0.5).add_to(map_osm)

			i = j
else:
	# (BORING) Plot polylines connecting nodes with simple straight lines:
	if (nn_cost > 0):
		# a) nearest neighbor:
		points = []
		for nodeID in nn_route:
			points.append(tuple([node[nodeID].latDeg, node[nodeID].lonDeg]))
		folium.PolyLine(points, color="red", weight=8.5, opacity=0.5).add_to(map_osm)
	if (ip_cost > 0):
		# b) ip:
		points = []
		for nodeID in ip_route:
			points.append(tuple([node[nodeID].latDeg, node[nodeID].lonDeg]))
		folium.PolyLine(points, color="green", weight=4.5, opacity=0.5).add_to(map_osm)
	if (sa_cost > 0):
		# c) simulate annealing:
		points = []
		for nodeID in sa_route:
			points.append(tuple([node[nodeID].latDeg, node[nodeID].lonDeg]))
		folium.PolyLine(points, color="blue", weight=6.5, opacity=0.5).add_to(map_osm)

map_osm.save(mapFile)

print "\nThe OSM map is saved in: %s" % (mapFile)

if (solveNN):
	print "\nNearest Neighbor Route:"
	print nn_route
	print "Nearest Neighbor 'Cost':"
	print nn_cost
if (solveIP):
	print "\nIP Route:"
	print ip_route
	print "IP 'cost':"
	print ip_cost
if (solveSA):
	print "\nSA Route:"
	print sa_route
	print "SA 'cost':"
	print sa_cost
