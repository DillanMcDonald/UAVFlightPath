import math as m
import numpy as np
from matplotlib import pyplot as plt


mapxdim = 1 #in kilometers
mapydim = 1 #in kilometers
#Assuming the flight is over a rectangular field
mapxdim = mapxdim*1000 #Convert to meters
mapydim = mapydim*1000 #Convert to meters
planecruise = 20 #this is knots
#planecruise = planecruise*0.514444 #conversion to m/s
start = [0][0] #starting position of plane on map
maxG = 2 #max number of G's in a turn
bankangle = 60 #bank angle in deg
bankangle = bankangle*(m.pi/180) #convert to radians
windspeed = 5 #this is mph
windspeed = windspeed*0.868976 #converted to knots
winddirection = 120 #this is deg from East to which the wind is pointing
winddirection = winddirection*(m.pi/180) #convert to radians
swathw = 100 #camera swathwidth in ft
swathw = swathw*0.3048 #conversion to m
wind = np.array([windspeed*m.cos(winddirection), windspeed*m.sin(winddirection)]) #break the windspeed into components
print('wind:',wind)
turnangle = 180 #define what a complete turn is
turnangle = turnangle*(m.pi/180) #convert to radians
planetruespeed = 0 #just a definition
magplanetruespeed = 0 #just a definition
completeturn = 0 # a boolean for completing a turn
rateofturn = 0 #just a definition
curplaneangl = 0 #the plane's angle

#test variables for position
xpos = [0]
ypos = [0]
interval = .000001 #time interval per iteration
time = 0 #in seconds
counter = 0
errormargin = .01 #define an error margin for the "vertical" distance calculations

#model a turn attempt number uno
turnstart = curplaneangl
while completeturn != 1:
    planetruespeed = np.array([(planecruise*m.cos(curplaneangl)), (planecruise*m.sin(curplaneangl))])+ wind #subract the wind from the true speed
    xpos.append(xpos[counter] + planetruespeed[0]*interval*1.68781) #new position in ft based on converted speed
    ypos.append(ypos[counter] + planetruespeed[1]*interval*1.68781) #new position in ft based on converted speed
    time = time + interval  # a counter in seconds
    magplanetruespeed = np.linalg.norm(planetruespeed) #get the magnitude of the true speed
    rateofturn = ((1091*m.tan(bankangle)*m.pi)/(magplanetruespeed*180)) #rads/s also assuming level flight
    curplaneangl = curplaneangl + rateofturn*interval #compute the current angle the plane is facing based on a second by second basis
    if curplaneangl <= ((m.pi/2) + errormargin) and curplaneangl >= ((m.pi/2) - errormargin):   #If loop documenting the horizontal distance covered before half the turn is complete
        turnwidth = xpos[counter+1]
    if curplaneangl >= (turnstart+turnangle): #exit condition loop
        completeturn = 1    #ofc if the turn is complete, exit the loop
    counter = counter + 1

#model a "straight" flight
completetravel = 0
curplanedist = xpos[counter] #document the current distance to indicate the end of the turn
diststart = curplanedist
distneeded = 100 #dist needed in ft
print(planetruespeed)

while completetravel != 1:
    xpos.append(xpos[counter] + planetruespeed[0]*interval*1.68781) #loop through the horizontal distance traveled
    ypos.append(ypos[counter] + planetruespeed[1]*interval*1.68781) #loop through the vertical distance traveled
    time = time + interval  # a counter in seconds
    counter = counter + 1
    curplanedist = xpos[counter]
    if distneeded <= abs(diststart-xpos[counter]): #
        completetravel = 1    #ofc if the turn is complete, exit the loop

#turn number 2
turnstart = curplaneangl
while completeturn != 1:
    planetruespeed = np.array([(planecruise*m.cos(curplaneangl)), (planecruise*m.sin(curplaneangl))])+ wind #subract the wind from the true speed
    xpos.append(xpos[counter] + planetruespeed[0]*interval*1.68781) #new position in ft based on converted speed
    ypos.append(ypos[counter] + planetruespeed[1]*interval*1.68781) #new position in ft based on converted speed
    time = time + interval  # a counter in seconds
    magplanetruespeed = np.linalg.norm(planetruespeed) #get the magnitude of the true speed
    rateofturn = -((1091*m.tan(bankangle)*m.pi)/(magplanetruespeed*180)) #rads/s also assuming level flight
    curplaneangl = curplaneangl + rateofturn*interval #compute the current angle the plane is facing based on a second by second basis
    if curplaneangl <= ((m.pi/2) + errormargin) and curplaneangl >= ((m.pi/2) - errormargin):   #If loop documenting the horizontal distance covered before half the turn is complete
        turnwidth = xpos[counter+1]
    if curplaneangl >= (turnstart-turnangle): #exit condition loop
        completeturn = 1 #ofc if the turn is complete, exit the loop
    counter = counter + 1


plt.plot(xpos, ypos)
plt.show()
print (turnwidth)


