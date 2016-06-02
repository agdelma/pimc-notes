'''
A path-integral quantum Monte Carlo program to compute the energy of the simple
harmonic oscillator in one spatial dimension.
'''

from __future__ import print_function
import numpy as np

# ------------------------------------------------------------------------------------------- 
def SHOEnergyExact(T):
    '''The exact SHO energy when \hbar \omega/ k_B = 1.''' 
    return 0.5/np.tanh(0.5/T)

# ------------------------------------------------------------------------------------------- 
def HarmonicOscillator(R):
    '''Simple harmonic oscillator potential with m = 1 and \omega = 1.'''
    return 0.5*np.dot(R,R);

# ------------------------------------------------------------------------------------------- 
class Paths:
    '''The set of worldlines, action and estimators.'''
    def __init__(self,beads,tau,lam):
        self.tau = tau
        self.lam = lam
        self.beads = np.copy(beads)
        self.numTimeSlices = len(beads)
        self.numParticles = len(beads[0])

    def SetPotential(self,externalPotentialFunction):
        '''The potential function. '''
        self.VextHelper = externalPotentialFunction

    def Vext(self,R):
        '''The external potential energy.'''
        return self.VextHelper(R)

    def PotentialAction(self,tslice):
        '''The potential action.'''
        pot = 0.0
        for ptcl in range(self.numParticles):
            pot += self.Vext(self.beads[tslice,ptcl]) 
        return self.tau*pot

    def KineticEnergy(self):
        '''The thermodynamic kinetic energy estimator.'''
        tot = 0.0
        norm = 1.0/(4.0*self.lam*self.tau*self.tau)
        for tslice in range(self.numTimeSlices):
            tslicep1 = (tslice + 1) % self.numTimeSlices
            for ptcl in range(self.numParticles):
                delR = self.beads[tslicep1,ptcl] - self.beads[tslice,ptcl]
                tot = tot - norm*np.dot(delR,delR)
        
        KE = 0.5*self.numParticles/self.tau + tot/(self.numTimeSlices)
        return KE

    def PotentialEnergy(self):
        '''The operator potential energy estimator.'''
        PE = 0.0
        for tslice in range(self.numTimeSlices):
            for ptcl in range(self.numParticles):
                R = self.beads[tslice,ptcl]
                PE = PE + self.Vext(R)
        return PE/(self.numTimeSlices)

    def Energy(self):
        '''The total energy.'''
        return self.PotentialEnergy() + self.KineticEnergy()

# ------------------------------------------------------------------------------------------- 
def PIMC(numSteps,Path):
    '''Perform a path integral Monte Carlo simulation of length numSteps.'''
    observableSkip = 50
    equilSkip = 1000
    numAccept = {'CenterOfMass':0,'Staging':0}
    EnergyTrace = []

    for steps in range(0,numSteps): 
        # for each particle try a center-of-mass move
        for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles):
                numAccept['CenterOfMass'] += CenterOfMassMove(Path,ptcl)

        # for each particle try a staging move
        for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles): 
            numAccept['Staging'] += StagingMove(Path,ptcl)

        # measure the energy
        if steps % observableSkip == 0 and steps > equilSkip:
            EnergyTrace.append(Path.Energy())
                     
    print('Acceptance Ratios:')
    print('Center of Mass: %4.3f' %
          ((1.0*numAccept['CenterOfMass'])/(numSteps*Path.numParticles)))
    print('Staging:        %1.3f\n' %
          ((1.0*numAccept['Staging'])/(numSteps*Path.numParticles)))
    return np.array(EnergyTrace)

# ------------------------------------------------------------------------------------------- 
def CenterOfMassMove(Path,ptcl):
    '''Attempts a center of mass update, displacing an entire particle
    worldline.'''
    delta = 0.75
    shift = delta*(-1.0 + 2.0*np.random.random())

    # Store the positions on the worldline
    oldbeads = np.copy(Path.beads[:,ptcl])

    # Calculate the potential action
    oldAction = 0.0
    for tslice in range(Path.numTimeSlices):
        oldAction += Path.PotentialAction(tslice)

    # Displace the worldline
    for tslice in range(Path.numTimeSlices):
        Path.beads[tslice,ptcl] = oldbeads[tslice] + shift

    # Compute the new action
    newAction = 0.0
    for tslice in range(Path.numTimeSlices):
        newAction += Path.PotentialAction(tslice)

    # Accept the move, or reject and restore the bead positions
    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        Path.beads[:,ptcl] = np.copy(oldbeads)
        return False

# ------------------------------------------------------------------------------------------- 
def StagingMove(Path,ptcl):
    '''Attempts a staging move, which exactly samples the free particle
    propagator between two positions.

    See: http://link.aps.org/doi/10.1103/PhysRevB.31.4234
    
    Note: does not work for periodic boundary conditions.
    '''

    # the length of the stage, must be less than numTimeSlices
    m = 16

    # Choose the start and end of the stage
    alpha_start = np.random.randint(0,Path.numTimeSlices)
    alpha_end = (alpha_start + m) % Path.numTimeSlices

    # Record the positions of the beads to be updated and store the action
    oldbeads = np.zeros(m-1)
    oldAction = 0.0
    for a in range(1,m):
        tslice = (alpha_start + a) % Path.numTimeSlices
        oldbeads[a-1] = Path.beads[tslice,ptcl]
        oldAction += Path.PotentialAction(tslice)

    # Generate new positions and accumulate the new action
    newAction = 0.0;
    for a in range(1,m):
        tslice = (alpha_start + a) % Path.numTimeSlices
        tslicem1 = (tslice - 1) % Path.numTimeSlices
        tau1 = (m-a)*Path.tau
        avex = (tau1*Path.beads[tslicem1,ptcl] +
                Path.tau*Path.beads[alpha_end,ptcl]) / (Path.tau + tau1)
        sigma2 = 2.0*Path.lam / (1.0 / Path.tau + 1.0 / tau1)
        Path.beads[tslice,ptcl] = avex + np.sqrt(sigma2)*np.random.randn()
        newAction += Path.PotentialAction(tslice)

    # Perform the Metropolis step, if we rejct, revert the worldline
    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        for a in range(1,m):
            tslice = (alpha_start + a) % Path.numTimeSlices
            Path.beads[tslice,ptcl] = oldbeads[a-1]
        return False

# ------------------------------------------------------------------------------------------- 
def main():
    T = 1.00  # temperature in Kelvin  
    lam = 0.5 # \hbar^2/2m k_B

    numParticles = 1    
    numTimeSlices = 20
    numMCSteps = 100000
    tau = 1.0/(T*numTimeSlices)
    binSize = 100

    print('Simulation Parameters:')
    print('N      = %d' % numParticles)
    print('tau    = %6.4f' % tau)
    print('lambda = %6.4f' % lam)
    print('T      = %4.2f\n' % T)

    # fix the random seed
    np.random.seed(1173)

    # initialize main data structure
    beads = np.zeros([numTimeSlices,numParticles])

    # random initial positions (classical state) 
    for tslice in range(numTimeSlices):
        for ptcl in range(numParticles):
            beads[tslice,ptcl] = 0.5*(-1.0 + 2.0*np.random.random())

    # setup the paths
    Path = Paths(beads,tau,lam)
    Path.SetPotential(HarmonicOscillator)

    # compute the energy via path-integral Monte Carlo
    Energy = PIMC(numMCSteps,Path)

    # Do some simple binning statistics
    numBins = int(1.0*len(Energy)/binSize)
    slices = np.linspace(0, len(Energy),numBins+1,dtype=int)
    binnedEnergy = np.add.reduceat(Energy, slices[:-1]) / np.diff(slices)

    # output the final result
    print('Energy = %8.4f +/- %6.4f' % (np.mean(binnedEnergy),
                                        (np.std(binnedEnergy)/np.sqrt(numBins-1))))
    print('Eexact = %8.4f' % SHOEnergyExact(T))

# ----------------------------------------------------------------------
if __name__ == "__main__": 
    main()
