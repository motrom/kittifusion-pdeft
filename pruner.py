#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last mod 9/14/19
"""
import numpy as np
import numba as nb
from singleIntegrator2 import ft_pexist as so_ft_pexist

#@nb.njit(nb.b1(nb.f8[:], nb.f8[:]))
#def objectsSame(obj1, obj2):
#    if abs(obj1[0]-obj2[0])>.1: return False
#    if abs(obj1[1]-obj2[1])>.1: return False
#    if abs(obj1[2]-obj2[2])>.1: return False
#    if abs(obj1[3]-obj2[3])>.2: return False
#    if abs(obj1[4]-obj2[4])>.1: return False
#    if abs(obj1[5]-obj2[5])>.01: return False
#    if abs(obj1[6]-obj2[6])>.3: return False
#    if abs(obj1[7]-obj2[7])>.5: return False
#    if abs(obj1[8]-obj2[8])>.5: return False
#    if abs(obj1[15]-obj2[15])>.5: return False
#    if abs(obj1[23]-obj2[23])>.1: return False
#    if abs(obj1[47]-obj2[47])>.01: return False
#    if abs(obj1[55]-obj2[55])>1.: return False
#    return True

@nb.njit(nb.b1(nb.f8[:], nb.f8[:]))
def objectsSame(obj1, obj2):
    if abs(obj1[0]-obj2[0])>.25: return False
    if abs(obj1[1]-obj2[1])>.25: return False
    if abs(obj1[2]-obj2[2])>.2: return False
    if abs(obj1[3]-obj2[3])>.5: return False
    if abs(obj1[4]-obj2[4])>.3: return False
    if abs(obj1[5]-obj2[5])>.02: return False
    if abs(obj1[6]-obj2[6])>.6: return False
    if abs(obj1[8]-obj2[8])>1: return False
    if abs(obj1[9]-obj2[9])>1: return False
    if abs(obj1[17]-obj2[17])>1: return False
    if abs(obj1[26]-obj2[26])>.2: return False
    if abs(obj1[53]-obj2[53])>.01: return False
    if abs(obj1[62]-obj2[62])>2.: return False
    return True
    
@nb.njit(nb.void(nb.f8[:,:], nb.b1[:,:], nb.f8[:]))
def pruneObjects(objects, associations, hypweights):
    #objectweights = objects[:,56] * objects[:,57] * np.dot(hypweights,associations)
    nobj = objects.shape[0]
    objectweights = np.zeros(nobj,dtype=np.float64)
    for objidx in range(nobj):
        #if np.any(associations[:,objidx]):
            #objectweight = objects[objidx,56]*objects[objidx,57]
            #objectweights *= np.sum(hypweights[associations[:,objidx]])
        if objects[objidx,so_ft_pexist] < 1e-4: continue
        for hypidx in range(associations.shape[0]):
            if associations[hypidx, objidx]:
                objectweights[objidx] = associations.shape[0]-hypidx
                break
    objorder = np.argsort(objectweights)[::-1]
    included = np.ones((nobj,), dtype=np.bool8)
    for i in range(nobj):
        objidx = objorder[i]
        if included[objidx]:
            for objidx2 in objorder[i+1:]:
                if included[objidx2] and objectsSame(objects[objidx], objects[objidx2]):
                    included[objidx2] = False
                    objects[objidx2,so_ft_pexist] = 0.
                    # replace with other object
                    associations[:,objidx] |= associations[:,objidx2]
                    associations[:,objidx2] = False
    
    
#def pruneHyps(objects, associations, hypweights):
#    objectsthatmatter = objects[:,so_ft_pexist] > 1e-3
#    hypothesesthatmatch = np.all(associations[:,objectsthatmatter] ==
#                                 associations[0,objectsthatmatter], axis=1)
#    hypothesesthatmatch[0] = False
#    hypweights[hypothesesthatmatch] = 1e3
#    
#    nexthyp = np.where(hypothesesthatmatch[1:]==False)[0]
#    if nexthyp.shape[0]>0:
#        nexthyp = nexthyp[0]
#        hypothesesthatmatch = np.all(associations[:,objectsthatmatter] ==
#                                     associations[nexthyp,objectsthatmatter], axis=1)
#        hypothesesthatmatch[:nexthyp+1] = False
#        hypweights[hypothesesthatmatch] = 1e3
        
@nb.njit(nb.void(nb.f8[:,:], nb.b1[:,:], nb.f8[:]))
def pruneHyps(objects, associations, hypweights):
    associations[:,objects[:,so_ft_pexist]<1e-4] = False
    #objectsthatmatter = np.where(objects[:,so_ft_pexist] > 1e-4)[0]
    nhyp, nobj = associations.shape
    for hyp in range(nhyp):
        if hypweights[hyp]==1e3: continue
        for hyp2 in range(hyp+1, nhyp):
            if hypweights[hyp2]==1e3: continue
            allmatch = True
            for objidx in range(nobj):
                if associations[hyp,objidx] != associations[hyp2,objidx]:
                    allmatch=False
                    break
            if allmatch:
                hypweights[hyp2] = 1e3