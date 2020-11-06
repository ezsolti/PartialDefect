#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions and variables for the MVA analysis

In case some other isotopes are of interest, the isotopes variable can be extended.

zs. elter 2020
"""

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import random
import os
import math
import re

from sklearn import preprocessing
from sklearn import decomposition

isotopes={'Cs134': {'hl':2.065*365,
                        'energies': [0.563,0.569,0.604,0.795,0.801,1.038,1.167,1.365],
                        'strength': [8.338,15.373,97.62,85.46,8.688,0.990,1.790,3.017]},
              'Cs137': {'hl':30.1*365,
                        'energies': [0.662],
                        'strength': [85.1]},
              'Eu154': {'hl':8.6*365,
                        'energies': [0.723,0.756,0.873,0.996,1.004,1.246,1.274,1.494,1.596],
                        'strength': [20.06,4.52,12.08,10.48,18.01,0.856,34.8,0.698,1.797]}}
              

def AssemblyMap(deftype):
    """Function to produce a 17x17 assembly map with partial defects in it.
    '1' represents fuel, '2' represent control rod guide, and '3' represents dummy rods.
    
    Parameters
    ----------
    deftype : str
       String variable to describe the assembly map (it takes values 'A', 'B', etc) 
    
    Returns
    -------
    mapArray : list of lists
        A list of list to represent a matrix describing the rod types in the assembly.
    """
    
    import random
    import math
    SubAsStr=''
    CrPos=[40,43,46,55,65,88,91,94,97,100,139,142,145,148,151,190,193,196,199,202,225,235,244,247,250]
    FuelPos=[]
    for i in range(17*17):
        j=i+1
        if j not in CrPos:        
            FuelPos.append(j)
    
    if deftype=='A':
        DummyPos=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,45,47,49,51,53,57,59,61,63,67,69,71,73,75,77,79,81,83,85,87,89,93,95,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,129,131,133,135,137,141,143,147,149,153,155,157,159,161,163,165,167,169,171,173,175,177,179,181,183,185,187,189,191,195,197,201,203,205,207,209,211,213,215,217,219,221,223,227,229,231,233,237,239,241,243,245,249,251,253,255,257,259,261,263,265,267,269,271,273,275,277,279,281,283,285,287,289]
    elif deftype=='B':
        DummyPos=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23,25,27,29,31,33,34,35,37,39,41,45,47,49,51,52,53,57,59,61,63,67,68,69,71,73,81,83,85,86,87,89,99,101,102,103,105,117,119,120,121,123,133,135,136,137,153,154,155,157,167,169,170,171,173,185,187,188,189,191,201,203,204,205,207,209,217,219,221,222,223,227,229,231,233,237,238,239,241,243,245,249,251,253,255,256,257,259,261,263,265,267,269,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289]
    elif deftype=='C':
        DummyPos=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,21,23,25,27,29,31,33,34,35,37,39,41,45,47,49,51,52,53,57,59,61,63,67,68,69,71,73,75,79,81,83,85,86,87,89,99,101,102,103,105,117,119,120,121,123,133,135,136,137,153,154,155,157,167,169,170,171,173,185,187,188,189,191,201,203,204,205,207,209,211,215,217,219,221,222,223,227,229,231,233,237,238,239,241,243,245,249,251,253,255,256,257,259,261,263,265,267,269,271,272,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288]
    elif deftype=='D':
        DummyPos=[2,3,4,5,6,7,8,10,11,12,13,14,15,16,18,21,23,25,27,29,31,34,35,37,39,41,45,47,49,51,52,53,57,59,61,63,67,68,69,71,73,75,79,81,83,85,86,87,89,99,101,102,103,105,107,109,113,115,117,119,120,121,123,133,135,136,154,155,157,167,169,170,171,173,175,177,181,183,185,187,188,189,191,201,203,204,205,207,209,211,215,217,219,221,222,223,227,229,231,233,237,238,239,241,243,245,249,251,253,255,256,259,261,263,265,267,269,272,274,275,276,277,278,279,280,282,283,284,285,286,287,288]
    elif deftype=='E':
        DummyPos=[22,24,25,27,28,30,37,49,58,59,61,62,70,73,75,76,78,79,81,84,104,106,107,109,110,112,113,115,116,118,121,123,124,126,127,129,130,132,133,135,155,157,158,160,161,163,164,166,167,169,172,174,175,177,178,180,181,183,184,186,206,209,211,212,214,215,217,220,228,229,231,232,241,253,260,262,263,265,266,268]
    elif deftype=='F':
        DummyPos=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,48,50,52,54,58,62,66,68,70,73,81,84,86,102,104,106,116,118,120,136,138,152,154,170,172,174,184,186,188,204,206,209,217,220,222,224,228,232,236,238,240,242,252,254,256,258,260,262,264,266,268,270,272,274,276,278,280,282,284,286,288]
    elif deftype=='G':
        DummyPos=[1,2,3,4,7,8,9,10,11,14,15,16,17,18,19,22,30,33,34,35,37,49,51,52,68,70,84,103,109,113,119,120,136,137,153,154,170,171,177,181,187,206,220,222,238,239,241,253,255,256,257,260,268,271,272,273,274,275,276,279,280,281,282,283,286,287,288,289]    
    elif deftype=='H':
        DummyPos=[1,2,4,6,8,9,10,12,14,16,17,18,20,24,28,32,34,36,50,52,58,62,68,73,81,86,102,104,106,116,118,120,136,137,153,154,170,172,174,184,186,188,204,209,217,222,228,232,238,240,254,256,258,262,266,270,272,273,274,276,278,280,281,282,284,286,288,289]
    elif deftype=='I':
        DummyPos=[1,2,3,5,8,9,10,13,15,16,17,18,19,33,34,35,51,56,64,69,75,79,85,120,136,137,153,154,170,205,211,215,221,226,234,239,255,256,257,271,272,273,274,275,277,280,281,282,285,287,288,289]  
    elif deftype=='J':
        DummyPos=[1,2,4,6,8,10,12,14,16,17,18,20,24,28,32,34,36,50,52,68,86,102,104,118,120,136,154,170,172,186,188,204,222,238,240,254,256,258,262,266,270,272,273,274,276,278,280,282,284,286,288,289]   
    elif deftype=='K':
        DummyPos=[1,2,3,5,8,10,13,15,16,17,18,19,33,34,35,51,69,85,120,136,154,170,205,221,239,255,256,257,271,272,273,274,275,277,280,282,285,287,288,289]  
    elif deftype=='L':
        DummyPos=[1,2,5,9,13,16,17,18,19,24,28,33,34,37,49,69,85,104,118,137,153,172,186,205,221,241,253,256,257,262,266,271,272,273,274,277,281,285,288,289]
    elif deftype=='M':
        DummyPos=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,51,52,68,69,85,86,102,103,119,120,136,137,153,154,170,171,187,188,204,205,221,222,238,239,255,256,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289]
    elif deftype=='N':
        DummyPos=[37,38,39,41,45,47,48,49,54,57,59,61,63,66,71,73,75,77,79,81,83,89,93,95,99,105,107,109,111,113,115,117,123,125,127,129,131,133,141,143,147,149,157,159,161,163,165,167,173,175,177,179,181,183,185,191,195,197,201,207,209,211,213,215,217,219,224,227,229,231,233,236,241,242,243,245,249,251,252,253]
    elif deftype=='O':
        DummyPos=[]
    elif deftype=='P':
        DummyPos=[37,38,39,41,42,54,56,57,58,59,60,71,72,73,74,75,76,77,89,90,92,93,105,106,107,108,109,110,111,122,123,124,125,126,127,128,140,141,143,144]
    elif deftype=='Q':
        DummyPos=[92,93,95,96,108,109,110,111,112,113,114,125,126,127,128,129,130,131,143,144,146,147,159,160,161,162,163,164,165,176,177,178,179,180,181,182,194,195,197,198]
    elif deftype=='R':
        DummyPos=[5,14,21,24,27,28,30,37,42,44,45,49,50,53,58,64,69,70,72,75,76,78,79,82,85,89,90,92,96,98,101,106,107,110,115,116,123,124,129,134,140,144,147,156,158,159,163,167,168,171,174,177,178,180,181,194,205,211,213,215,216,219,222,224,232,236,237,239,240,246,257,260,261,268,276,278,282,283,285,286]
    elif deftype=='S':
        DummyPos=[1,2,3,4,5,6,7,8,9,18,19,20,21,22,23,24,25,35,36,37,38,39,41,42,52,53,54,56,57,58,59,60,69,70,71,72,73,74,75,76,86,87,89,90,92,93,103,104,105,106,107,108,109,110,111,120,121,122,123,124,125,126,127,138,141,144,146,149,152,163,164,165,166,167,168,169,170,179,180,181,182,183,184,185,186,187,197,198,200,201,203,204,214,215,216,217,218,219,220,221,230,231,232,233,234,236,237,238,248,249,251,252,253,254,255,265,266,267,268,269,270,271,272,281,282,283,284,285,286,287,288,289]
    elif deftype=='T':
        DummyPos=[1,2,3,4,5,6,7,8,9,18,19,20,21,22,23,24,25,35,36,37,38,39,41,42,52,53,54,56,57,58,59,60,69,70,71,72,73,74,75,76,86,87,89,90,92,93,103,104,105,106,107,108,109,110,111,120,121,122,123,124,125,126,127,137,138,140,141,143,144,154,155,156,157,158,159,160,161,162,171,172,173,174,175,176,177,178,188,189,191,192,194,195,205,206,207,208,209,210,211,212,213,222,223,224,226,227,228,229,239,240,241,242,243,245,246,256,257,258,259,260,261,262,263,264,273,274,275,276,277,278,279,280]
    ND=len(DummyPos)
    mapArray=[]
    col=1
    for i in range(17*17):
        j=i+1
        if col==1:
            SubAsStr=SubAsStr+'        '
            mapArray.append([])
        if j in CrPos:
            SubAsStr=SubAsStr+' '+str(3)#str(random.choice([4,6]))
            mapArray[-1].append('3')
        elif j in DummyPos:
            SubAsStr=SubAsStr+' '+str(2)
            mapArray[-1].append('2')
        else:
            SubAsStr=SubAsStr+' '+str(1)
            mapArray[-1].append('1')
        if col==17:
            SubAsStr=SubAsStr+'\n'
            col=0
        col=col+1
    SubAsStr=SubAsStr[:-1]
        
    return mapArray


def detectorEff(E):
    """Function to describe the detector efficiency. It is based on a fit of 
    Serpent2 results.
    
    Parameters
    ----------
    E : float or list
        Energy in MeV
    
    Returns
    -------
    Eps : float or list
        Detector efficiency at energy/energies E
    """
    E=E*1000 #change MeV to keV
    a=-8.02741343e-02
    b=-1.49151904e-01
    c=-2.84160334e-01
    d=4.39778388e-02
    e=1.19674986e-03
    f=1.26102944e+02
    
    lnEps=a+b*np.log(E/f)+c*(np.log(E/f))**2+d*(np.log(E/f))**3+e*(np.log(E/f))**4
    
    Eps=np.exp(lnEps)
    return Eps
              
def gammaLines(row,nuclides=['Cs137','Cs134','Eu154']):
    """
    Function to return the energy line intensities. For this the concentrations of nuclides
    are converted into activity concentrations, and then into the intensity of the lines. This
    value is basically in emission/cm3/s units.
    
    Parameters
    ----------
    row : dict or pandas dataframe row
        dictionary to keep track of the spent fuel inventory.
    nuclides : list
        list of the nuclide identifiers for which the gamma line intensities are to be evaluated
    
    Returns
    -------
    lines : dict
        nested dictionary to store the gamma line intensities. outer keys are nuclide identifiers,
        inner keys are 'energies' and 'strength'. 
    energies : numpy array
        List of gamma line energies.
    """
    d2s=86400
    
    lines={}
    for iso in isotopes:
        lines[iso]={'energies':[],'strength':[]}
        conci=row[iso]
        acti=conci*1e24*(np.log(2)/(isotopes[iso]['hl']*d2s))
        for en,br in zip(isotopes[iso]['energies'],isotopes[iso]['strength']):
            enfreq=acti*br
            lines[iso]['energies'].append(en)
            lines[iso]['strength'].append(enfreq)
    
    energies=[]
    for key in nuclides:
        for en in isotopes[key]['energies']:
            energies.append(key+': '+str(en))
    energies=np.array(energies)
    return lines,energies


def prepareX(fuellib=None, cases=['O','R'],nuclides=['Cs137','Cs134','Eu154'],ratio=False,scaling=True,normalization=True,
             encoding={'O':0,'R':1},gefffilestem='outs/geomEff_',fresh=False):
    """
    Function to create the X matrix, which has the feature vectors as its rows.
    This is not a very straightforward function, it is just to wrap up some data 
    management.
    
    Parameters
    ----------
    fuellib : pandas dataframe
        Fuel library. For further details see https://doi.org/10.1016/j.dib.2020.106429
    cases : list of strings
        The assembly types in the analysis. It is important that the geometric efficiency files
        include these strings in it.
    ratio : bool
        if True the pairwise ratios of the features are returned.
    scaling : bool
        if True standard scaling is performed on the data
    normalization : bool
        if True, the feature vectors are normalized to sum to 1
    encoding : dictionary
        For classification numeric labels are preferred, thuse the cases need to be ecoded.
        Here one can also make sure whether several cases should be encoded into only 2 classes.
    gefffilestem : str
        The path and filename stem of the geometric efficiency curves. This will be 
        extended with the string describing the case.
    fresh : bool
        If True, always the geometric efficiency of fresh fuel is used.
        
    Returns
    -------
    data : numpy array
        The data matrix. If the ratios are requested, then the matrix of the ratios
    labels : numpy array
        Encoded labels for all samples
    energies : numpy array
        Array of energies or energie ratios to identify the columns of the data matrix
    """
    if fuellib is None:
        raise TypeError('No fuel library is added.')
    colN=0
    for nucl in nuclides:
        colN=colN+len(isotopes[nucl]['energies'])
    
    data=np.empty((0,colN))
    dataratio=np.empty((0,int((colN**2-colN)/2)))
    labels=[]    
    
    
    for index, fuel in fuellib.iterrows():
        lines,energies=gammaLines(fuel)
        peaks=['']
        for c in cases:
            if fresh:
                engeff=np.loadtxt(gefffilestem+'%s_orig.dat'%c)
            else:
                engeff=np.loadtxt(gefffilestem+'%s_%d.dat'%(c,fuel['id']))
            peaks={}
            for iso in lines:
                peaks[iso]={'energies': [], 'strength':[]}
                for en,st in zip(lines[iso]['energies'],lines[iso]['strength']):
                    peaks[iso]['energies'].append(en)
                    peaki=st*np.interp(en,engeff[:,0],engeff[:,1])*detectorEff(en)
                    peaks[iso]['strength'].append(peaki)

            feature=[]
            for nucl in nuclides:
                for strength in peaks[nucl]['strength']:
                    feature.append(strength)
            feature=np.array(feature)
            
            featureratio=[]
            energiesratio=[]
            for i,(fi,ei) in enumerate(zip(feature,energies)):
                for j,(fj,ej) in enumerate(zip(feature,energies)):
                    if j>i:
                        if fi/fj >= fj/fi:
                            featureratio.append(fi/fj)
                        else:
                            featureratio.append(fj/fi)
                        energiesratio.append(ei+' / '+ej)
            featureratio=np.array(featureratio)

            if normalization:
                feature=feature/sum(feature)
                featureratio=featureratio/sum(featureratio)
            
            data=np.vstack([data,feature])
            dataratio=np.vstack([dataratio,featureratio])
            
            labels.append(encoding[c]) 
    #TODO bring normalization here, to see the impact of doing scaling first
    labels=np.array(labels)
    
    if scaling:
        data = preprocessing.scale(data)
        dataratio = preprocessing.scale(dataratio)
    
    if ratio:
        return dataratio,labels,np.array(energiesratio)
    else:
        return data,labels,energies

def AtConc_to_ZWeightPer(row):
    """Function to convert spent fuel inventory from nuclidewise atom concentration
    to elementwise weight percentage. 
    The function expects a pandas row from the MVA dataset.
    1, calculates the mass per cm3 for each isotope
    2, changes the ZAID into elements and sums the mass of each element
    3, returns the normalized mass (ie w%) for each element
    
    Parameters
    ----------
    row : pandas dataframe
        Row in dataframe describing the fuel inventory
    
    Returns
    -------
    masspervolume : dict
        Dictionary which stores the mass percentage for each element (which are the keys).
    """
    
    #precondition row into dictionary
    inventory = row.to_dict()
    inventory.pop('Unnamed: 0') #note this is highly specific
    inventory.pop('BU')
    inventory.pop('CT')
    inventory.pop('IE')
    inventory.pop('fuelType')
    inventory.pop('TOT_SF')
    inventory.pop('TOT_GSRC')
    inventory.pop('TOT_A')
    inventory.pop('TOT_H')
    
    NA = 6.022140857E23
    masspervolume = {}
    
    
    for iso in inventory:
        isoText = re.findall('\D+', iso)
        isoNum =  re.findall('\d+', iso)
        Z=isoText[0]
        A=float(isoNum[0])
            
        massconci = A*((inventory[iso]*1e24)/NA)  #this gives the mass of that isotope in g/cm3
        
        if Z in masspervolume:
            masspervolume[Z]=masspervolume[Z]+massconci
        else:
            masspervolume[Z]=massconci
            
        
    #getting weight%
    summass=sum(masspervolume.values())
    for element in masspervolume:
        masspervolume[element]=masspervolume[element]/summass
    return masspervolume
    

    
def XCOMmaterial(massdic):
    """
    Function to convert the mass percentages into XCOM readable input string
    
    Parameters
    ----------
    massdic : dict
        Dictionary which stores the mass percentage for each element (which are the keys).
    
    Returns
    -------
    xcomstr : str
        String which includes the elementwise mass percentages in an XCOM readable form.
    """
    #printf 'spentfuel\n4\n2\nH\n0.1\nO\n0.9\n1\n3\n1\n3\n0.6\n0.8\n0.9\nN\ntestauto.out\n1\n' | ./XCOMtest
    xcomstr=''    
    for element in massdic:
        xcomstr=xcomstr+element+'\n'+str(massdic[element])+'\n'
    return xcomstr
    

#Variable to match element symbol to Z. not used in these functions!
nametoZ= {'Mt': 109,
          'Hs': 108,
          'Bh': 107,
          'Sg': 106,
          'Db': 105,
          'Rf': 104,
          'Lr': 103,
          'No': 102,
          'Md': 101,
          'Fm': 100,
          'Es': 99,
          'Cf': 98,
          'Bk': 97,
          'Cm': 96,
          'Am': 95,
          'Pu': 94,
          'Np': 93,
          'U': 92,
          'Pa': 91,
          'Th': 90,
          'Ac': 89,
          'Ra': 88,
          'Fr': 87,
          'Rn': 86,
          'At': 85,
          'Po': 84,
          'Bi': 83,
          'Pb': 82,
          'Tl': 81,
          'Hg': 80,
          'Au': 79,
          'Pt': 78,
          'Ir': 77,
          'Os': 76,
          'Re': 75,
          'W': 74,
          'Ta': 73,
          'Hf': 72,
          'Lu': 71,
          'Yb': 70,
          'Tm': 69,
          'Er': 68,
          'Ho': 67,
          'Dy': 66,
          'Tb': 65,
          'Gd': 64,
          'Eu': 63,
          'Sm': 62,
          'Pm': 61,
          'Nd': 60,
          'Pr': 59,
          'Ce': 58,
          'La': 57,
          'Ba': 56,
          'Cs': 55,
          'Xe': 54,
          'I': 53,
          'Te': 52,
          'Sb': 51,
          'Sn': 50,
          'In': 49,
          'Cd': 48,
          'Ag': 47,
          'Pd': 46,
          'Rh': 45,
          'Ru': 44,
          'Tc': 43,
          'Mo': 42,
          'Nb': 41,
          'Zr': 40,
          'Y': 39,
          'Sr': 38,
          'Rb': 37,
          'Kr': 36,
          'Br': 35,
          'Se': 34,
          'As': 33,
          'Ge': 32,
          'Ga': 31,
          'Zn': 30,
          'Cu': 29,
          'Ni': 28,
          'Co': 27,
          'Fe': 26,
          'Mn': 25,
          'Cr': 24,
          'V': 23,
          'Ti': 22,
          'Sc': 21,
          'Ca': 20,
          'K': 19,
          'Ar': 18,
          'Cl': 17,
          'S': 16,
          'P': 15,
          'Si': 14,
          'Al': 13,
          'Mg': 12,
          'Na': 11,
          'Ne': 10,
          'F': 9,
          'O': 8,
          'N': 7,
          'C': 6,
          'B': 5,
          'Be': 4,
          'Li': 3,
          'He': 2,
          'H': 1}