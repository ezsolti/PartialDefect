
import numpy as np
import os
import math
import re


def AtConc_to_ZWeightPer(row):
    """the function expects a pandas row from the MVA dataset.
    1, calculates the mass per cm3 for each isotope
    2, changes the ZAID into elements and sums the mass of each element
    3, returns the normalized mass (ie w%) for each element"""
    
    #precondition row into dictionary
    inventory = row.to_dict()
    inventory.pop('Unnamed: 0')
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
    #printf 'spentfuel\n4\n2\nH\n0.1\nO\n0.9\n1\n3\n1\n3\n0.6\n0.8\n0.9\nN\ntestauto.out\n1\n' | ./XCOMtest
    xcomstr=''    
    for element in massdic:
        xcomstr=xcomstr+element+'\n'+str(massdic[element])+'\n'
    return xcomstr
    


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