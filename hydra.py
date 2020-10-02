'''Author: Kostas Batsis
Supervised by: Ot de Wiljes
Description: An integrate-and-fire based model of the Hydra species nerve net.'''

from pandas import *
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
import pyspike as psp
from brian2 import *
prefs.codegen.target = 'cython'
start_scope()

def wellProbability(edgesProb):
    '''Creates a well shaped probability distribution
    edgesProb is the probability at each of the two edges of the well
    NOTE: edgesProb values must be greater than 0.15 and with a maximum of 0.5'''
    if(edgesProb > 0.15 and edgesProb <= 0.5):
        middleProb = 1-(edgesProb*2)
        distributionValues = arange(0,1,0.001)
        pValues = repeat(edgesProb/150,150)
        pValues = append(pValues,repeat(middleProb/700,700))
        pValues = append(pValues,repeat(edgesProb/150,150))        
        return choice(a=distributionValues,size=1,p=pValues)
    else:
        raise ValueError('edgesProb must greater than 0.15 and with a maximum of 0.5')

def createColumn(neuronalGroup,populationSize,columnLength):
    """Creates a cylindrical column of neurons
    neuronalGroup is a NeuronGroup object
    populationSize is the number of neurons in the column
    columnLength is the length of the column in arbitrary units
    NOTE: requires scipy.spatial/distance"""
    #adds attribute for storing the position vectors of the neurons
    neuronalGroup.add_attribute('positionVectors')
    #initialising the attribute as an array
    neuronalGroup.positionVectors = zeros((populationSize,3))
    #variables for the current provisional coordinates    
    xcor = 0.0
    ycor = 0.0
    zcor = 0.0 #z is the longitudinal axis    
       
    for i in range(0,populationSize,1):
        #while loop compares distances between neurons to make sure they are
        #not too close to one another
        tooClose = True
        while tooClose == True: 
            tooClose = False
            #creation of random coordinates
            zcor = (wellProbability(0.21)*columnLength) #random length value
            angle = (rand()*(2*pi))  #random angle
            xcor = (cos(angle)) #unpacks x coordinate from angle
            ycor = (sin(angle)) #likewise for y 
             
            if i == 0: # in this case just store the cooordinates                             
                neuronalGroup.positionVectors[i][0] = xcor
                neuronalGroup.positionVectors[i][1] = ycor
                neuronalGroup.positionVectors[i][2] = zcor            
            else: 
                #otherwise compare euclidean distances with all existing neurons
                #minimum allowed distance depends on the z-axis position of the 
                #current neuron's provisional coordinates
                currentVector = [xcor,ycor,zcor] #current neuron's position vector
                for j in range(i):
                    euDistance = distance.euclidean(neuronalGroup.positionVectors[j],currentVector)                    
                    if currentVector[2] > 1.5 and currentVector[2] < 8.5:
                        if abs(euDistance) < 0.2:                     
                            tooClose = True #neurons too close, while loop will try again
                            break
                    elif abs(euDistance) < 0.1:                     
                        tooClose = True 
                        break
                if tooClose == False:
                    #if distance ok stores coordinates                    
                    neuronalGroup.positionVectors[i][0] = xcor
                    neuronalGroup.positionVectors[i][1] = ycor                 
                    neuronalGroup.positionVectors[i][2] = zcor
    return neuronalGroup
    
def plotColumn(neuronalGroup,synapseGroup=None):
    """Creates a 3D scatterplot of the column with lines indicating the
    synaptic connections (if available)
    neuronalGroup is a NeuronGroup object with coordinates created by 
    createColumn()
    synapseGroup (optional) is a Synapses object""" 
    pV = neuronalGroup.positionVectors   
    figure1 = figure('Column')
    axes3d = figure1.add_subplot(111, projection='3d')
    axes3d.set_xlim(-1.5,1.5)
    axes3d.set_ylim(-1.5,1.5)
    axes3d.set_zlim(0,11)
    axes3d.scatter(xs=pV[:,0],ys=pV[:,1],zs=pV[:,2])
    axes3d.set_xlabel('x')
    axes3d.set_ylabel('y')
    axes3d.set_zlabel('z')
    
    if(synapseGroup != None): #plots the synaptic connections
        synArray = array([synapseGroup.i,synapseGroup.j])
        synArray = synArray.T
        for i in range(len(synArray)):
            axes3d.plot(xs=[pV[synArray[i,0],0],pV[synArray[i,1],0]],
                        ys=[pV[synArray[i,0],1],pV[synArray[i,1],1]],
                        zs=[pV[synArray[i,0],2],pV[synArray[i,1],2]],color='r')
                
def createSynapses(neuronalGroup,populationSize,synapticWeights,pValue,synapticDelay):
    '''Creates the synaptic connections in a column
    neuronalGroup is a NeuronGroup object
    populationSize is the number of neurons in the column
    synapticWeight is the weight of the synapses
    pValue is the probability of creating each synapse
    NOTE: requires scipy.spatial/distance'''
    if(populationSize > 1):
        #Creates the object that contains the synapses and their operations
        S = Synapses(neuronalGroup,neuronalGroup,'w:1',on_pre='v_post += w',
                     delay=synapticDelay*ms)
        currentNeuron = 0
        pV = neuronalGroup.positionVectors
    
    #while loop creates a bidirectional synapse if the distance between two
    #neurons is smaller than a specified distance that depends on the z-axis 
    #position of the current neuron
        while currentNeuron < (populationSize-1):
            for neighbor in range(currentNeuron+1,populationSize,1):
                euDistance = distance.euclidean(pV[neighbor],pV[currentNeuron])    
                if pV[currentNeuron,2] > 1.5 and pV[currentNeuron,2] < 8.5:
                    if abs(euDistance) < 0.5:
                        S.connect(i=currentNeuron,j=neighbor,p=pValue)
                        S.connect(i=neighbor,j=currentNeuron,p=pValue)
                elif abs(euDistance) < 0.3:
                    S.connect(i=currentNeuron,j=neighbor,p=pValue)
                    S.connect(i=neighbor,j=currentNeuron,p=pValue)
            currentNeuron += 1
        
        S.w = synapticWeights #sets the synaptic weights   
        return S
    else:
        print('createSynapses: Population size must be greater than 1')

def fftSignal(spikeTimes,runTimeTotal,adjustmentTime):
    '''Creates a signal suitable for FFT analysis
    spikesTimes: a SpikeMonitor.t object
    runTimeTotal: the total time of the simulation
    adjustmentTime: the time given to the network for adjustment
    NOTE: runTime should be a power of two
    NOTE: requires itertools/groupby'''
    #creates a dictionary with the times where firings occured as keys and 
    #the counts of the firings at each of these times
    spikeTimes = array(spikeTimes/second)
    spikeTimes = list(spikeTimes)
    spikeTimes.sort()
    times = set(spikeTimes)
    times = list(times)
    for i in range(len(times)):
        times[i] = round(times[i],3) 
    times.sort()
    for i in range(len(spikeTimes)):
        spikeTimes[i] = round(spikeTimes[i],3) 
    freqs = []
    for i in times:
        freqs.append(spikeTimes.count(i))
    spikes = dict(zip(times,freqs))      

    #fills the spikes dictionary with the rest of the zero valued times and
    #the associated zero valued counts
    timeArray = [x * 0.001 for x in range(adjustmentTime,runTimeTotal)]
    for i in range(len(timeArray)):
        timeArray[i] = round(timeArray[i],3)
    for i in timeArray:
        if i not in spikes:
            spikes[i] = 0

    #sorts and seperates the keys and the values and returns a list containing
    #them
    times = []
    freqs = []
    for key in sorted(spikes.iterkeys()):
        times.append(key)
        freqs.append(spikes[key])
    return([times,freqs])
        
def fftAnalysis(freqs,sRate):
    '''FFT analysis, returns the frequency bins and the associated power
    freqs: an array with the counts of firings at each sampling point
    sRate: the sampling rate based on the defaultclock.dt value 
    NOTE: length of freqs should be a power of two
    code based on: https://plot.ly/matplotlib/fft/'''
    Fs = float(sRate);  #sampling rate
    Ts = 1.0/Fs; #sampling interval

    n = len(freqs) #length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T #two sides frequency range
    frq = frq[range(n/2)] #one side frequency range

    Y = fft(freqs)/n #fft computing and normalization
    Y = Y[range(n/2)]

    figure1 = figure('Spectrum')
    fplot = figure1.add_subplot(111)
    fplot.plot(frq,abs(Y),'r') #plotting the spectrum
    fplot.set_xlabel('Freq (Hz)')
    fplot.set_ylabel('|Y(freq)|')
    show('Spectrum')
    return([frq,abs(Y)])
        
###################################################################
populationSize = 880 #number of neurons
columnLength = 10 #column length in arbitrary units
Vt = float64(0.998690173613014) #threshold for neuron firing
#synapticWeights = 0.15 #weight of synaptic connections
#synapticDelay = 2 #value of synaptic delay (in ms)
defaultclock.dt = 1*ms #sets the time step for the simulation(s)
runTime = 1800000 #duration of the to be analysed simulation (in ms)


eqs = '''
dv/dt = (I-v)/tau : 1 (unless refractory)
I : 1
tau : second
'''

synwArray = arange(0.1,1,0.1)
synwArray = append(synwArray,Vt)
cl = ['tau','delay','p(syn)','weight','SPIKEd','ISId','SPIKEsync']
qvalues = DataFrame(zeros((400,7)),columns=cl)
currentLine = 0
for tc in [70000]:
    for delay in (range(2,9,2)):
        for psyn in arange(0.1,1.1,0.1):
            for synw in synwArray:
                G1 = NeuronGroup(populationSize,eqs,threshold='v>Vt',reset='v=0',refractory=20*ms)
                G1.I = 1
                G1.tau = tc*ms
                G1.v = 'rand()*Vt'
                G1 = createColumn(G1,populationSize,columnLength)
                S1 = createSynapses(G1,populationSize,synw,psyn,delay)
                Sp1 = SpikeMonitor(G1)

                run(runTime*ms)

                #code below calculates and stores the pyspike metrics
                firingValuesWithUnits = Sp1.spike_trains().values()
                firingValues = []
                for i in range(len(firingValuesWithUnits)):
                    firingValues.append(array(firingValuesWithUnits[i])) 
                fV = open('fv.txt','w')
                for item in firingValues:
                    item = (" ".join(map(str,item)))
                    fV.write("%s\n" % item)
                fV.close()
                spikeTrains = psp.load_spike_trains_from_txt("fv.txt",edges=(0,runTime/1000.0))
                qvalues.iloc[currentLine,0] = tc
                qvalues.iloc[currentLine,1] = delay
                qvalues.iloc[currentLine,2] = psyn
                qvalues.iloc[currentLine,3] = synw
                qvalues.iloc[currentLine,4] = psp.spike_distance(spikeTrains)
                qvalues.iloc[currentLine,5] = psp.isi_distance(spikeTrains)
                qvalues.iloc[currentLine,6] = psp.spike_sync(spikeTrains)
                currentLine += 1 

                del G1
                del S1
                del Sp1
                del firingValuesWithUnits
                del firingValues
                del spikeTrains
     
qvalues.to_excel('qvalues.xlsx', sheet_name='Sheet1')