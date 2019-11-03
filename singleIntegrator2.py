#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last mod 9/7/19
handles things considered by the iPDA, or multi bernoulli filters
In this case,
    object existence - bernoulli probability
    object genuity
    object detectability x 8 - one for each camera and lidar
    object previous position x - for determining long-term false positives
    object previous position y - " "
    current number of camera detections
    current number of lidar box detections
    current number of lidar frag detections
    average score of camera detections
    label integer, (most likely) unique for each originating msmt
    object previous orientation cos
    object previous orientation sin
    stationaryness (previously directly in genuity score)
"""
import numpy as np
from segment import lasers2use
from singleTracker import prepSample as soPrepObject
from singleTracker import prepMeasurement as soPrepMeasurement
from singleTracker import likelihood as soLikelihood
from singleTracker import update as soUpdate
from singleTracker import likelihoodNewObject as soLikelihoodNewObject
from singleTracker import mlSample as soNewObject
from singleTracker import validSample as soValidSample
from singleTracker import predict as soPredict
from singleTracker import reOrient as soReOrient
from singleTracker import positionDistribution as soPositionDistribution
from singleTracker import report as soReport
from singleTracker import nft as sonft

NOFAKES = False
NODETECT = False

nlasers = len(lasers2use)
ft_pexist = sonft
ft_genuity = sonft+1
ft_detectability_cam = sonft+2
ft_detectability_lidar = sonft+3
ft_oldpos_x = sonft+3+nlasers
ft_oldpos_y = sonft+4+nlasers
ft_nmsmts_cam = sonft+5+nlasers
ft_nmsmts_lidarbox = sonft+6+nlasers
ft_nmsmts_lidarfrag = sonft+7+nlasers
ft_avgscore_cam = sonft+8+nlasers
ft_label = sonft+9+nlasers
ft_thisdetect = sonft+10+nlasers
ft_oldori_x = sonft+11+nlasers
ft_oldori_y = sonft+12+nlasers
ft_genuitymod = sonft+13+nlasers
nft = sonft + 14 + nlasers

ft_detectabilities = {'lidar{:d}'.format(i):ft_detectability_lidar+i
                       for i in range(nlasers)}
ft_detectabilities['camera'] = ft_detectability_cam

survival_probability = .997
# average probability of object being detected
detectability_steadystate = np.array([.99] + [.98]*nlasers)
detectability_timestep_ratio = .75 # amount it changes in one timestep
if NODETECT:
    detectability_timestep_ratio = .99999
# log of rate of new objects, effectively multiplier on occupancy grid
newobject_rate_const = {'camera':-4., 'lidarbox':-2., 'lidarfrag':0.5}
# lidar measurements don't have quality scores -- set to .5 for no effect
# for camera, genuity ratio warps initial genuity (not just set to score)
# this doesn't actually matter anymore, could be removed
newobject_genuity_ratio = {'camera':.5, 'lidarbox':.1, 'lidarfrag':.001}
# reach of discounted moving average
oldobj_discountrate = .8

# fake objects less likely to survive -- effectively mixing in clutter
# this is on top of survival_probability
fakeobjects_survival = .99


detectability_pos2pos = 1 - (1 - detectability_steadystate) * detectability_timestep_ratio
detectability_neg2pos = detectability_steadystate * detectability_timestep_ratio
detectability_times = detectability_pos2pos - detectability_neg2pos
detectability_plus = detectability_neg2pos
def predict(sample):
    soPredict(sample[:sonft])
    pexist = sample[ft_pexist]
    preal = sample[ft_genuity]
    oldposx, oldposy = sample[[ft_oldpos_x, ft_oldpos_y]]
    posx, posy = sample[:2]
    newpexist = pexist*survival_probability
    # remove stuff outside of kitti visible zone
    if posx < 0 or posx > 55 or abs(posy) > posx*.87+1.87:
        newpexist *= .5
    if posx < -5 or posx > 63 or abs(posy) > posx*.87+6:
        newpexist = 0.
    # change genuity based on absolute and relative motion
    sample[ft_genuitymod] = 1.
    # extra survival (mixing) for fake objects
    avgscore = sample[ft_avgscore_cam] 
    avgscore = avgscore / (avgscore + (1-avgscore)*fakeobjects_survival)
    sample[ft_avgscore_cam] = avgscore
    # adding in modifier
    newpreal = avgscore
    # existence is sum of exist*genuine and exist*fake
    # #P(exist, genuine) is unchanged
    newpexist *= fakeobjects_survival if newpreal == 0 else preal / newpreal 
    sample[ft_pexist] = newpexist
    sample[ft_genuity] = newpreal
    # mix all detectabilities
    sample[ft_detectability_cam:ft_detectability_cam+nlasers+1] *= detectability_times
    sample[ft_detectability_cam:ft_detectability_cam+nlasers+1] += detectability_plus
    # discounted moving average for previous positions
    currdist = np.hypot(sample[0], sample[1])
    sample[ft_oldpos_x] = oldposx*oldobj_discountrate + posx*(1-oldobj_discountrate)
    sample[ft_oldpos_y] = oldposy*oldobj_discountrate + posy*(1-oldobj_discountrate)
    neworix = (sample[ft_oldori_x]*oldobj_discountrate +
                                   posx/currdist*(1-oldobj_discountrate))
    neworiy = (sample[ft_oldori_y]*oldobj_discountrate +
                                   posy/currdist*(1-oldobj_discountrate))
    oricorrection = np.hypot(neworix, neworiy)
    sample[ft_oldori_x] = neworix/oricorrection
    sample[ft_oldori_y] = neworiy/oricorrection
    

    
def logOdds(num,den):
    return -25. if num < 1e-11 else 25. if den < 1e-11 else np.log(num/den)
    
# logit(k * logOdds(p))
def bernoulliWarp(p, k):
    return 0. if p<1e-11 else 1. if 1-p<1e-11 else 1/(1+(1/p-1)**k)

"""
returns:
    prepped object from single tracker
    negative log-odds of match (for adding to NLL of match matrix)
"""
def prepObject(sample, visibility, sensorname):
    soprep = soPrepObject(sample[:sonft], sensorname)
    ft_detectability = ft_detectabilities[sensorname]
    pexist, preal, pdetect = sample[[ft_pexist, ft_genuity, ft_detectability]]
    pmatch = pexist*pdetect
    punmatch = 1 - pexist*pdetect*visibility
    podds = -logOdds(pmatch, punmatch)
    if sensorname == 'camera' and sample[ft_nmsmts_cam]>0:
        discount = min(9, sample[ft_nmsmts_cam])
        weightedscore = bernoulliWarp(preal, discount/(discount+1))
    else:
        weightedscore = preal
    return soprep + (weightedscore, podds)

""" 9/15/19 """ 
def prepObjectCorrector(sample, visibility, sensorname):
    ft_detectability = ft_detectabilities[sensorname]
    pexist, preal, pdetect = sample[[ft_pexist, ft_genuity, ft_detectability]]
    punmatch = 1 - pexist*pdetect*visibility
    return logOdds(1., punmatch)

"""
returns:
    prepped measurement from single tracker
    negative log-odds of msmt match (for adding to NLL of match matrix)
    unlike VJ tracking, fake objects are disentangled from the newobjectrate
"""
def prepMeasurement(msmt, newobjectrate, sensorname):
    if sensorname == 'camera':
        score = msmt[5]
        sosensorname = 'camera'
    elif type(msmt[0]) == np.ndarray:
        sosensorname = 'lidarbox'
        score = .5
    else:
        sosensorname = 'lidarfrag'
        score = .5
    soprep = soPrepMeasurement(msmt, sosensorname)
    newmsmtlik = soLikelihoodNewObject(msmt, sosensorname)
    llnewmsmt = np.log(newobjectrate) + newobject_rate_const[sosensorname]
    return soprep + (score, llnewmsmt - newmsmtlik)

"""
weights are for determining which objects to prune or report
only their relative values matter, but these are from 0 to 1
using probability of existence * probability of genuity
"""
def postMatchWeight(sample, msmt):
    return sample[ft_genuity]

def postObjMissWeight(sample, visibility, sensorname):
    ft_detectability = ft_detectabilities[sensorname]
    pexist, preal, pdetect = sample[[ft_pexist, ft_genuity, ft_detectability]]
    existdetect = pdetect * visibility
    pexist = pexist*(1-existdetect) / (pexist*(1-existdetect) + 1-pexist)
    return pexist * preal

" returning initial pexist * estimated genuity (which is just msmt score) "
def postMsmtMissWeight(msmt, newobjectrate, sensorname):
    if sensorname == 'camera':
        return msmt[5]
    else:
        return 0.

"""
actually negative log likelihood
this already accounts for false positive and negative probabilities
score term is 0 for fully matched or uncertain objects, up to infinite for completely off
"""
def likelihood(preppedsample, preppedmsmt):
    genuity, objlogterm = preppedsample[-2:]
    score, misslogterm = preppedmsmt[-2:]
    scorell = -np.log(genuity*score + (1-genuity)*(1-score))
    solik = soLikelihood(preppedsample[:-2], preppedmsmt[:-2])
    return solik + scorell + objlogterm + misslogterm


def updateMatch(sample, msmt, sensorname):
    newsample = sample.copy()
    if sensorname == 'camera':
        sosensorname = 'camera'
    elif type(msmt[0]) == np.ndarray:
        sosensorname = 'lidarbox'
    else:
        sosensorname = 'lidarfrag'
    ft_detectability = ft_detectabilities[sensorname]
    newsample[:sonft] = soUpdate(sample[:sonft], msmt, sosensorname)
    oldpexist = sample[ft_pexist]
    newsample[ft_pexist] = 1.
    newsample[ft_detectability] = 1.
    if sosensorname == 'camera':
        # maintain (moving) average score
        # update genuity to reflect change
        score = msmt[5]
        if NOFAKES:
            score = 1.
        npreviousdets = sample[ft_nmsmts_cam]
        meanscore = sample[ft_avgscore_cam]
        meanscore = logOdds(meanscore, 1-meanscore)
        score = logOdds(score, 1-score)
        avgweight = min(npreviousdets, 9)
        newmeanscore = (meanscore*avgweight+score)/(avgweight+1)
        newmeanscore = 1/(1+np.exp(-newmeanscore))
        newsample[ft_genuity] = newmeanscore / (newmeanscore+(1-newmeanscore)*
                                                 sample[ft_genuitymod])
        newsample[ft_nmsmts_cam] += 1
        newsample[ft_avgscore_cam] = newmeanscore
    elif sosensorname == 'lidarbox':
        if sample[ft_nmsmts_lidarbox] == 0:
            pass # do something to make genuity more likely??
        newsample[ft_nmsmts_lidarbox] += 1
    else:
        newsample[ft_nmsmts_lidarfrag] += 1
        ### this is very wrong... but might fix the issue of fragments allowing
        # false msmts from boxes/camera to exist indefinitely
        newsample[ft_pexist] = oldpexist
    return newsample
    
" modify features of object that was not measured this step "
def updateMiss(sample, visibility, sensorname):
    newsample = sample.copy()
    ft_detectability = ft_detectabilities[sensorname]
    pexist, preal, pdetect = sample[[ft_pexist, ft_genuity, ft_detectability]]
    existdetect = pdetect * visibility
    newsample[ft_pexist] = pexist*(1-existdetect) / (1-pexist*existdetect)
    newsample[ft_detectability] = pdetect*(1-visibility) / (1-existdetect)
    return newsample

""" assign features to new object
"""
def updateNew(msmt, newobjectrate, sensorname):
    newsample = np.zeros(nft)
    if sensorname == 'camera':
        sosensorname = 'camera'
        score = msmt[5]
        genuity_ratio = newobject_genuity_ratio['camera']
        ft_nmsmts = ft_nmsmts_cam
    elif type(msmt[0]) == np.ndarray:
        sosensorname = 'lidarbox'
        score = .5
        genuity_ratio = newobject_genuity_ratio['lidarbox']
        ft_nmsmts = ft_nmsmts_lidarbox
    else:
        sosensorname = 'lidarfrag'
        score = .5
        genuity_ratio = newobject_genuity_ratio['lidarfrag']
        ft_nmsmts = ft_nmsmts_lidarfrag
    newsample[:sonft] = soNewObject(soPrepMeasurement(msmt, sosensorname))
    newsample[ft_pexist] = 1.
    llexistgenu = score * genuity_ratio
    llexist = llexistgenu + (1-score)*(1-genuity_ratio)
    newsample[ft_genuity] = llexistgenu / llexist
    newsample[ft_avgscore_cam] = llexistgenu / llexist
    if NOFAKES:
        newsample[ft_pexist] = llexistgenu / llexist
        newsample[ft_genuity] = 1.
        newsample[ft_avgscore_cam] = 1.
    newsample[ft_detectability_cam:ft_detectability_cam+nlasers+1] = detectability_steadystate
    newsample[ft_detectabilities[sensorname]] = 1.
    newsample[ft_nmsmts] = 1
    newsample[ft_label] = np.random.randint(int(1e9))
    posx,posy = newsample[:2]
    newsample[ft_oldpos_x] = posx
    newsample[ft_oldpos_y] = posy
    newsample[ft_oldori_x] = posx/np.hypot(posx,posy)
    newsample[ft_oldori_y] = posy/np.hypot(posx,posy)
    newsample[ft_genuitymod] = 1.
    if sosensorname != 'camera':
        newsample[ft_pexist] = 0.
    return newsample

def validSample(sample):
    valid = True
    if sample[ft_pexist] > 1e-2:
        valid &= soValidSample(sample[:sonft])
    elif sample[ft_pexist] > 1e-5 and not soValidSample(sample[:sonft]):
        print("invalid (but very unlikely) sample")
    probability_features = [ft_pexist, ft_genuity, ft_avgscore_cam]
    probability_features += range(ft_detectability_cam,ft_detectability_cam+nlasers+1)
    valid &= np.all(sample[probability_features] >= 0)
    valid &= np.all(sample[probability_features]-1 <= 1e-10)
    sample[probability_features] = np.minimum(sample[probability_features], 1)
    return valid

def reOrient(sample, newpose):
    soReOrient(sample, newpose)
    # don't reorient oldpos --- this is used to examine how object's view changes
    # reorient oldori, which is used to examine change to angle of view
    sample[[ft_oldori_x,ft_oldori_y]] = np.dot(newpose[:2,:2],
                                              sample[[ft_oldori_x,ft_oldori_y]])
    
positionDistribution = soPositionDistribution

""" returns a float that is the prune weight, and the object in report format """
def report(sample):
    return sample[ft_pexist]*sample[ft_genuity], soReport(sample[:sonft])

""" if sample has 0 pexist, no reason to perform calculations (might be empty) """
def shouldUseObject(sample): return sample[ft_pexist]*sample[ft_genuity] > 1e-10