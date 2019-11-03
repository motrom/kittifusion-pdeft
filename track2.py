# -*- coding: utf-8 -*-
"""
last modified 9/7/19
"""

import numpy as np
import numba as nb
np.random.seed(0)
from cv2 import imshow, waitKey, destroyWindow
from imageio import imread, get_writer
from time import time

from presavedSensor import getMsmts as getCamMsmts
from plotStuff import plotImgKitti, addRect2KittiImg, hsv2Rgba, plotRectangleEdges
from plotStuff import plotPoints
from calibs import calib_extrinsics, calib_projections, view_by_day
from trackinginfo import sceneranges
from trackinginfo import calib_map_training as calib_map
from analyzeGT import readGroundTruthFileTracking#, formatForKittiScoreTrack
from config import grndlen, grndstart, grndstep, grnd2checkgrid, floor
from occupancygrid import reOrientGrid, mixGrid, mapNormal2Subgrid
import singleIntegrator2 as singleIntegrator
soPrepObject = singleIntegrator.prepObject
soLikelihood = singleIntegrator.likelihood
soUpdateMatch = singleIntegrator.updateMatch
soUpdateMiss = singleIntegrator.updateMiss
soUpdateNew = singleIntegrator.updateNew
soPostMatchWeight = singleIntegrator.postMatchWeight
soPostObjMissWeight = singleIntegrator.postObjMissWeight
soPostMsmtMissWeight = singleIntegrator.postMsmtMissWeight
soPrepMsmt = singleIntegrator.prepMeasurement
validSample = singleIntegrator.validSample
soPredict = singleIntegrator.predict
soReOrient = singleIntegrator.reOrient
soPositionDistribution = singleIntegrator.positionDistribution
soReport = singleIntegrator.report
shouldUseObject = singleIntegrator.shouldUseObject
so_ft_pexist = singleIntegrator.ft_pexist
n_ft = singleIntegrator.nft
soPrepCorrector = singleIntegrator.prepObjectCorrector
from selfpos import loadSelfTransformations
from mhtdaClink import mhtda, allocateWorkvarsforDA, sparsifyByRow
from mhtdaClink import processOutput as mhtdaProcessOutput
from occlusion import pointCloud2OcclusionImg, occlusionImg2Grid
from hyphopping import HypHopLabeler # swapLabelsForHopping
from segment import processLidar, lasers2use, groundforlaser_angles, laser_angles
from occlusion_old import occlude2 as occlude
from pruner import pruneObjects, pruneHyps


lidar_files = '/home/m2/Data/kitti/tracking_velodyne/training/{:04d}/{:06d}.bin'
img_files = '/home/m2/Data/kitti/tracking_image/training/{:04d}/{:06d}.png'
gt_files = '/home/m2/Data/kitti/tracking_gt/{:04d}.txt'
oxt_files = '/home/m2/Data/kitti/oxts/{:04d}.txt'
output_img_files = '../tracking/measurements/a/{:04d}/{:06d}.png'
ground_files = '/home/m2/Data/kitti/tracking_ground/training/{:02d}f{:06d}.npy'
outestimatefiles = '/home/m2/Data/kitti/estimates/trackingresultsMGLmht/{:02d}f{:04d}.npy'
videofile = None#'resultsScene4.mp4'
scene_idx = 2

n_msmts_max = 100
n_objects = 400
n_hyps = 250#

# visibility map
# update every timestep
# new obj prob map -- previous turns' visibility and new objs
# uncertainty map -- for each object in tile: exist prob, real prob, and variance
# propagate and update occupancy grid, given measurements and occlusion
#occupancy_mixer = np.array([[.02, .02, .02],
#                            [.02, .83, .02],
#                            [.02, .02, .02]])
occupancy_mixer = np.array([[.01, .04, .01],
                            [.04, .75, .04],
                            [.01, .04, .01]])
occupancy_mixer_laser = np.array([[.005, .02, .005],
                                  [ .02,  .9,  .02],
                                  [.005, .02, .005]])
occupancy_constant_add = .05
occupancy_outer = 1. # starting new object rate
occupancydummy = np.zeros((grndlen[0]+occupancy_mixer.shape[0]-1,
                           grndlen[1]+occupancy_mixer.shape[1]-1))


def mhtdaPruneOutput(assocs, assocspruned, updatepairs, updatepairspruned,
                     prepruneweights, hypweights, nsols, nin, nout):
    hypweights2 = np.exp(hypweights[0] - hypweights[:nsols])
    prepruneweights[:nin] *= np.dot(hypweights2, assocs[:nsols,:nin])
    keepidxs = np.argpartition(prepruneweights[:nin], nin-nout)[nin-nout:]
    assocspruned[:] = False
    assocspruned[:nsols,:nout] = assocs[:nsols,keepidxs]
    updatepairspruned[:nout] = updatepairs[keepidxs]
    # just for debugging with score outside
    prepruneweights[:nout] = prepruneweights[keepidxs]
    
#def beforeOcclusion(positiondist, beforedetectzone):
#    xpos,ypos = positiondist[:2]
#    objorientation = np.arctan2(ypos,xpos)
#    detectzoneidx = np.searchsorted(groundforlaser_angles, objorientation)
#    if detectzoneidx > 0 and detectzoneidx < beforedetectzone.shape[0]:
#        visibledist = beforedetectzone[detectzoneidx-1]
#        objdist = np.hypot(ypos,xpos)
#        return max(.25, min(1, objdist/visibledist))
#    return 1.

def beforeOcclusion(positiondist, ground):
    xpos,ypos = positiondist[:2]
    grndx = max(min(int(xpos/grndstep[0])-grndstart[0], grndlen[0]-1), 0)
    grndy = max(min(int(ypos/grndstep[1])-grndstart[1], grndlen[1]-1), 0)
    grndelev = (ground[grndx,grndy,3] - ground[grndx,grndy,0]*xpos
                                      - ground[grndx,grndy,1]*ypos)
    heightoflaser = 1.65-grndelev + laser_angles[laser2use]*np.hypot(xpos,ypos)
    return (1. if heightoflaser < 1.1 else
            .7 if heightoflaser < 1.5 else 
            .3 if heightoflaser < 2 else .1)


def gtCenters():
    return [gtobj['box'][:2] for gtobj in gt]
def nearCenter(x,y):
    return np.where((np.hypot(objects[:,0]-x,objects[:,1]-y) < 3) &
                    (objects[:,56] > .001))[0]


startfileidx, endfileidx = sceneranges[scene_idx]
#startfileidx = 110
#endfileidx = 5
calib_idx = calib_map[scene_idx]
calib_extrinsic = calib_extrinsics[calib_idx].copy()
calib_extrinsic[2,3] += 1.65
view_angle = view_by_day[calib_idx]
view_for_lidar = view_angle + .1
calib_projection = calib_projections[calib_idx] ## dot extrinsic ???
with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
gt_all, gtdontcares = readGroundTruthFileTracking(gtfilestr, ('Car', 'Van'))
selfpos_transforms = loadSelfTransformations(oxt_files.format(scene_idx))

# initialize state objects
objects = np.zeros((n_objects, n_ft))
associations = np.zeros((n_hyps, n_objects), dtype=np.bool8)
hypweights = np.zeros(n_hyps)
statebeforeupdate = (objects, associations, hypweights)
newobjects = objects.copy()
newassociations = associations.copy()
newhypweights = hypweights.copy()
stateafterupdate = (newobjects, newassociations, newhypweights)

n_objects_pre_prune = n_objects*4 + n_msmts_max*4
matches = np.zeros((n_objects, n_msmts_max))
msmtsubset = np.zeros((1,n_msmts_max), dtype=np.bool8)
msmtsubsetweights = np.zeros(1)
updatepairs = np.zeros((n_objects, 2), dtype=int)
updatepairspreprune = np.zeros((n_objects_pre_prune, 2), dtype=int)
associationspreprune = np.zeros((n_hyps, n_objects_pre_prune), dtype=np.bool8)
mhtdaworkvars = allocateWorkvarsforDA(n_objects, n_msmts_max, n_hyps)
mhtdaprocessindex = np.zeros((n_objects+1, n_msmts_max+1), dtype=int)
association_pairs = np.zeros((n_hyps, n_objects+n_msmts_max, 2), dtype=np.int32)
prepruneweights = np.zeros(n_objects_pre_prune)
nvalidhypotheses = 1 # initial hypothesis, no objects

objectdetectprobs = np.zeros(n_objects)
occupancy = np.zeros(grndlen) + occupancy_outer
occupancy_laser = occupancy.copy()
occupancy_transform = np.eye(3)
tilescores = occupancy.copy()
visibility = occupancy.copy()
occlusionimg = None
motionreports = []
hyphoplabeler = HypHopLabeler(n_objects, 200, 10)
updatetime = 0.
minhypgap = 100.

if videofile is not None: video = get_writer(videofile, mode='I', fps=4)

for fileidx in range(startfileidx, endfileidx):
    # get data for this timestep
    data = np.fromfile(lidar_files.format(scene_idx, fileidx),
                       dtype=np.float32).reshape((-1,4))[:,:3]
    #data = data.dot(calib_extrinsic[:3,:3].T) + calib_extrinsic[:3,3]
    img = imread(img_files.format(scene_idx, fileidx))[:,:,::-1]
    gt = gt_all[fileidx]
    selfposT = selfpos_transforms[fileidx][[0,1,3],:][:,[0,1,3]]
    ground = np.load(ground_files.format(scene_idx, fileidx))
    
    # propagate objects
    for objidx in range(n_objects):
        obj = objects[objidx]
        if not shouldUseObject(obj): continue
        soReOrient(obj, selfposT)
        if obj[8] > 50 or obj[17] > 50:
            #assert obj[42]*obj[43] < .1
            if obj[so_ft_pexist] > .01:
                print("removed variant object at {:.0f},{:.0f} w/ exist {:.2f}".format(
                        obj[0], obj[1], obj[so_ft_pexist]))
            # too big to characterize well
            # just treat as poisson and add to occupancy
            tilex, tiley = np.floor(obj[:2]/grndstep).astype(int)-grndstart
            splituppexist = obj[so_ft_pexist]/25.
            occupancy[max(tilex-2,0):tilex+3, max(tiley-2,0):tiley+3] += splituppexist
            obj[so_ft_pexist] = 0.
        soPredict(obj)
    hypweights -= np.min(hypweights) # prevent endless decrease in weights
    
    # propagate new object zones
    occupancy_transform = selfposT.dot(occupancy_transform)
    if abs(occupancy_transform[0,2]) > 1.2:
        occupancy = reOrientGrid(occupancy, occupancy_transform, occupancy_outer,
                                 grndstep, grndstart, grndlen)
        occupancy_laser = reOrientGrid(occupancy_laser, occupancy_transform,
                                       occupancy_outer, grndstep, grndstart, grndlen)
        occupancy_transform = np.eye(3)
    # mix nearby tiles
    mixGrid(occupancy, occupancy_mixer, occupancy_outer, occupancydummy)
    occupancy += occupancy_constant_add
    occupancy[0,15:17] = 0. # this is the host car
    mixGrid(occupancy_laser, occupancy_mixer_laser, occupancy_outer, occupancydummy)
    
    ## CAMERA
    sensorname = 'camera'
    cammsmts = getCamMsmts(scene_idx, fileidx)
    msmts = cammsmts
    nmsmts = len(msmts)
    
    # determine occlusion and emptiness
    datacalib = data.dot(calib_extrinsic[:3,:3].T) + calib_extrinsic[:3,3]
    occlusionimg = pointCloud2OcclusionImg(datacalib, occlusionimg)
    occlusionImg2Grid(occlusionimg, visibility, ground)
    # b.c. you don't actually have full lidar to use for occlusion
    # TODO replace with non-lidar visibility
    visibility = np.maximum(visibility, .04)
    
    # estimate detection probability based on visibility
    # probably need two different object detection probabilities
    for objidx in range(n_objects):
        if shouldUseObject(objects[objidx]):
            positiondist = soPositionDistribution(objects[objidx])
            subgridloc, occupysubgrid = mapNormal2Subgrid(positiondist,
                                            grndstart,grndstep,grndlen, subsize=2)
            subgridend = subgridloc + occupysubgrid.shape
            visibilitysubgrid = visibility[subgridloc[0]:subgridend[0],
                                           subgridloc[1]:subgridend[1]]
            objectdetectprobs[objidx] = np.einsum(occupysubgrid, [0,1],
                                                  visibilitysubgrid, [0,1], [])
        else:
            objectdetectprobs[objidx] = 0. # shouldn't matter...
    assert np.all(objectdetectprobs < 1+1e-8)
    objectdetectprobs = np.minimum(objectdetectprobs, 1)
    
    # prepare objs/msmts for data association
    # first get occupancy in area of msmts
    msmtsprepped = []
    occupancyatmsmts = []
    for msmt in msmts:
        tilex, tiley = floor(msmt[:2]/grndstep).astype(int)-grndstart
        occupancyatmsmt = occupancy[tilex, tiley]
        occupancyatmsmts.append(occupancyatmsmt)
        msmtsprepped.append(soPrepMsmt(msmt, occupancyatmsmt, sensorname))
    # data association
    for objidx in range(n_objects):
        if not shouldUseObject(objects[objidx]):
            matches[objidx,:nmsmts] = 100
            continue
        objectprepped = soPrepObject(objects[objidx], objectdetectprobs[objidx],
                                     sensorname)
        for msmtidx, msmt in enumerate(msmtsprepped):
            matches[objidx,msmtidx] = soLikelihood(objectprepped, msmt)
    msmtsubset[0,:nmsmts] = True
    msmtsubset[0,nmsmts:] = False
    matches2 = matches[:,:nmsmts]
    if matches2.shape[1]>0:
        matchorder = np.argsort(np.min(matches2, axis=1))
    else:
        matchorder = np.arange(matches.shape[0])
    matches2 = sparsifyByRow(matches2[matchorder], 4)#matches#
    associations2 = associations[:,matchorder].copy()
    # 9/15/19 huge deal, adding obj miss terms to hypweights
    objmissterms = np.array([soPrepCorrector(objects[objidx],
                                             objectdetectprobs[objidx], sensorname)
                             for objidx in range(n_objects)])
    hypweights += associations.dot(objmissterms)
    starttime = time()
    mhtda(matches2, associations2, hypweights, nvalidhypotheses,
          msmtsubset, msmtsubsetweights, association_pairs, newhypweights,
          mhtdaworkvars)
    updatetime += time() - starttime
    nvalidhypotheses = sum(newhypweights < 1000)
    for hypidx in range(nvalidhypotheses):
        association_pairs_to_move = association_pairs[hypidx,:,0] >= 0
        association_pairs[hypidx,association_pairs_to_move,0] = matchorder[
                        association_pairs[hypidx,association_pairs_to_move,0]]
    nupdatepairspreprune = mhtdaProcessOutput(updatepairspreprune,
                                              associationspreprune,
                                              association_pairs[:nvalidhypotheses],
                                              mhtdaprocessindex, n_objects_pre_prune)
    assert mhtdaprocessindex[-1,-1] == -1
    nupdatepairs = min(n_objects, nupdatepairspreprune)
    for newobjidx in range(nupdatepairspreprune):
        objidx, msmtidx = updatepairspreprune[newobjidx]
        if objidx == -1:
            prepruneweight = soPostMsmtMissWeight(msmts[msmtidx],
                                                  occupancyatmsmts[msmtidx],
                                                  sensorname)
        elif msmtidx == -1:
            prepruneweight = soPostObjMissWeight(objects[objidx],
                                                 objectdetectprobs[objidx],
                                                 sensorname)
        else:
            prepruneweight = soPostMatchWeight(objects[objidx], msmts[msmtidx])
        prepruneweights[newobjidx] = prepruneweight
    mhtdaPruneOutput(associationspreprune, newassociations, updatepairspreprune,
                     updatepairs, prepruneweights, newhypweights,
                     nvalidhypotheses, nupdatepairspreprune, nupdatepairs)
    # tentative update
    for newobjidx in range(nupdatepairs):
        objidx, msmtidx = updatepairs[newobjidx]
        if objidx == -1:
            newobjects[newobjidx] = soUpdateNew(msmts[msmtidx],
                                      occupancyatmsmts[msmtidx], sensorname)
        elif msmtidx == -1:
            newobjects[newobjidx] = soUpdateMiss(objects[objidx],
                                      objectdetectprobs[objidx], sensorname)
        else:
            msmt = msmtsprepped[msmtidx]
            newobjects[newobjidx] = soUpdateMatch(objects[objidx], msmts[msmtidx],
                                                  sensorname)
    # resolve update
    for objidx in range(nupdatepairs):
        if shouldUseObject(newobjects[objidx]):
            assert validSample(newobjects[objidx])
    #    if updatepairs[objidx,1] >= 0:
    #        assert newobjects[objidx,5] > .45 and newobjects[objidx,5] < 2.1  
    newobjects[nupdatepairs:,so_ft_pexist] = 0. # make sure unused pairs don't remain
    # object pruning 9/14/19
    if fileidx % 3:
        pruneObjects(newobjects, newassociations, newhypweights)
    # MHT pruning 9/3/19
    pruneHyps(newobjects, newassociations, newhypweights)
    
    statebeforeupdate, stateafterupdate = (stateafterupdate, statebeforeupdate)
    objects, associations, hypweights = statebeforeupdate
    newobjects, newassociations, newhypweights = stateafterupdate
    if nvalidhypotheses > 3:
        assert np.ptp(hypweights) > 1e-10
    
    occupancy *= 1-visibility
    
    
        
    ## LIDAR
    allboxes = []
    allfrags = []
    for laser2useidx in range(len(lasers2use)):#range(len(lasers2use)-1,-1,-1):
        laser2use = lasers2use[laser2useidx]
        sensorname = 'lidar{:d}'.format(laser2useidx)
        processedlidar = processLidar(data, calib_extrinsic, ground,
                                      view_for_lidar, laser2use)
        beforedetectzone, occlusionmap, boxmsmts, fragmsmts = processedlidar
        
        # estimate detection probability based on visibility
        for objidx in range(n_objects):
            if shouldUseObject(objects[objidx]):
                # also using minimal detection distance
                # make sure this doesn't lower objectdetectprobs so much
                # that objects go unmatched
                positiondist = soPositionDistribution(objects[objidx])
                assert positiondist[2]>1e-10 and positiondist[3]>1e-10
                # need to uncalibrate bc occlusion map is in lidar pov
                mean4occlusion = objects[objidx,:5].copy()
                mean4occlusion[:2] = np.linalg.inv(calib_extrinsic[:2,:2]).dot(
                                        mean4occlusion[:2] - calib_extrinsic[:2,2])
                objocclude = occlude(mean4occlusion,
                                     objects[objidx,8:72].reshape((8,8)),
                                     occlusionmap)
                #objocclude = occlude(positiondist, occlusionmap)
                objectdetectprobs[objidx] = 1-objocclude
#                objectdetectprobs[objidx] *= beforeOcclusion(positiondist,
#                                 beforedetectzone)
                objectdetectprobs[objidx] *= beforeOcclusion(positiondist, ground)
            else:
                objectdetectprobs[objidx] = 0. # shouldn't matter...
        assert np.all(objectdetectprobs < 1+1e-8)
        objectdetectprobs = np.minimum(objectdetectprobs, 1)
                    
        # process msmts, get relevant info (occupancy)
        msmts = []
        msmtsprepped = []
        occupancyatmsmts = []
        for msmt in boxmsmts:
            tilex, tiley = floor(msmt[0][:2]/grndstep).astype(int)-grndstart
            if tilex < 1 or tilex >= grndlen[0] or tiley < 0 or tiley >= grndlen[1]:
                continue
            msmts.append(msmt)
            occupancyatmsmt = occupancy_laser[tilex, tiley]
            occupancyatmsmts.append(occupancyatmsmt)
            msmtprepped = soPrepMsmt(msmt, occupancyatmsmt, sensorname)
            msmtsprepped.append(msmtprepped)
            allboxes.append(msmt[0])
        lastfrag = (-100.,0.)
        for msmt in fragmsmts:
            tilex, tiley = floor(msmt[:2]/grndstep).astype(int)-grndstart
            if tilex < 0 or tilex >= grndlen[0] or tiley < 0 or tiley >= grndlen[1]:
                continue
            if np.hypot(msmt[0]-lastfrag[0],msmt[1]-lastfrag[1]) < .4:
                continue
            msmts.append(msmt)
            occupancyatmsmt = occupancy_laser[tilex, tiley]
            occupancyatmsmts.append(occupancyatmsmt)
            msmtprepped = soPrepMsmt(msmt, occupancyatmsmt, sensorname)
            msmtsprepped.append(msmtprepped)
            allfrags.append(msmt)
            lastfrag = msmt
        nmsmts = len(msmts)
        assert nmsmts <= n_msmts_max
        # prep for data association
        for objidx in range(n_objects):
            if not shouldUseObject(objects[objidx]):
                matches[objidx,:nmsmts] = 100
                continue
            objectprepped = soPrepObject(objects[objidx], objectdetectprobs[objidx],
                                         sensorname)
            for msmtidx, msmt in enumerate(msmtsprepped):
                matches[objidx,msmtidx] = soLikelihood(objectprepped, msmt)
        # data association
        msmtsubset[0,:nmsmts] = True
        msmtsubset[0,nmsmts:] = False
        matches2 = matches[:,:nmsmts]
        if matches2.shape[1]>0:
            matchorder = np.argsort(np.min(matches2, axis=1))
        else:
            matchorder = np.arange(matches.shape[0])
        matches2 = sparsifyByRow(matches2[matchorder], 3)#matches#
        associations2 = associations[:,matchorder].copy()
        # 9/15/19 huge deal, adding obj miss terms to hypweights
        objmissterms = np.array([soPrepCorrector(objects[objidx],
                                                 objectdetectprobs[objidx], sensorname)
                                 for objidx in range(n_objects)])
        hypweights += associations.dot(objmissterms)
        starttime = time()
        mhtda(matches2, associations2, hypweights, nvalidhypotheses,
              msmtsubset, msmtsubsetweights, association_pairs, newhypweights,
              mhtdaworkvars)
        updatetime += time() - starttime
        nvalidhypotheses = sum(newhypweights < 1000)
        for hypidx in range(nvalidhypotheses):
            association_pairs_to_move = association_pairs[hypidx,:,0] >= 0
            association_pairs[hypidx,association_pairs_to_move,0] = matchorder[
                            association_pairs[hypidx,association_pairs_to_move,0]]
        nupdatepairspreprune = mhtdaProcessOutput(updatepairspreprune,
                                                  associationspreprune,
                                                  association_pairs[:nvalidhypotheses],
                                                  mhtdaprocessindex, n_objects_pre_prune)
        assert mhtdaprocessindex[-1,-1] == -1
        nupdatepairs = min(n_objects, nupdatepairspreprune)
        for newobjidx in range(nupdatepairspreprune):
            objidx, msmtidx = updatepairspreprune[newobjidx]
            if objidx == -1:
                prepruneweight = soPostMsmtMissWeight(msmts[msmtidx],
                                                      occupancyatmsmts[msmtidx],
                                                      sensorname)
            elif msmtidx == -1:
                prepruneweight = soPostObjMissWeight(objects[objidx],
                                                     objectdetectprobs[objidx],
                                                     sensorname)
            else:
                prepruneweight = soPostMatchWeight(objects[objidx], msmts[msmtidx])
            prepruneweights[newobjidx] = prepruneweight
        mhtdaPruneOutput(associationspreprune, newassociations, updatepairspreprune,
                         updatepairs, prepruneweights, newhypweights,
                         nvalidhypotheses, nupdatepairspreprune, nupdatepairs)
        # tentative update
        for newobjidx in range(nupdatepairs):
            objidx, msmtidx = updatepairs[newobjidx]
            if objidx == -1:
                newobjects[newobjidx] = soUpdateNew(msmts[msmtidx],
                                          occupancyatmsmts[msmtidx], sensorname)
            elif msmtidx == -1:
                newobjects[newobjidx] = soUpdateMiss(objects[objidx],
                                          objectdetectprobs[objidx], sensorname)
            else:
                msmt = msmtsprepped[msmtidx]
                newobjects[newobjidx] = soUpdateMatch(objects[objidx], msmts[msmtidx],
                                                      sensorname)
        # resolve update
        for objidx in range(nupdatepairs):
            if shouldUseObject(newobjects[objidx]):
                assert validSample(newobjects[objidx])
        # make sure unused pairs don't remain
        newobjects[nupdatepairs:,so_ft_pexist] = 0.
        # MHT pruning 9/3/19
        if n_hyps > 1: minhypgap = min(minhypgap, max(newhypweights[1:])-newhypweights[0])
        pruneHyps(newobjects, newassociations, newhypweights)
        if nvalidhypotheses > 3:
            assert np.ptp(newhypweights) > 1e-10
        statebeforeupdate, stateafterupdate = (stateafterupdate, statebeforeupdate)
        objects, associations, hypweights = statebeforeupdate
        newobjects, newassociations, newhypweights = stateafterupdate
        
        
        
    # update laser occupancy once, with all msmts
    # for NOFAKES, only use the last two lines!
    for msmt in allboxes:
        tilex, tiley = floor(msmt[:2]/grndstep).astype(int)-grndstart
        occupancy_laser[tilex, tiley] += .5
    for msmt in allfrags:
        tilex, tiley = floor(msmt[:2]/grndstep).astype(int)-grndstart
        occupancy_laser[tilex, tiley] += .5
    occupancy_laser *= .5
    occupancy_laser.clip(.05, 1., out=occupancy_laser)    
    
    # report
    reportedobjects = []
    reportedscores = []
    reportedlabels = []
    reportedidxs = []
    for objidx in range(n_objects):
        if associations[0,objidx]:
            reportscore, reportobj = soReport(objects[objidx])
            if reportscore > .1:
                reportedidxs.append(objidx)
                reportedobjects.append(reportobj)
                reportedscores.append(reportscore)
                reportedlabels.append(objects[objidx,so_ft_pexist+9+len(lasers2use)])
#                if objects[objidx,57]>.4:
#                    assert objects[objidx,5] > .66 and objects[objidx,5] < 1.5
    reportedobjects = np.array(reportedobjects).reshape((-1,5))
    reportedscores = np.array(reportedscores)
    reportedlabels = np.array(reportedlabels)
    fullreports = np.concatenate((reportedobjects, reportedscores[:,None],
                                  reportedlabels[:,None]), axis=1)
    ### label retargeting
    #fullreports[:,6] = hyphoplabeler.add(updatepairs, reportedidxs)
    np.save(outestimatefiles.format(scene_idx, fileidx), fullreports)
        
#    # check out motion stats for not-real objects
#    # TODO option 1
#    # save all objects, updatepairs, hypotheses
#    # afterwards, backtrack best hypotheses and match to gt
#    # find false objects and where they stop existing
#    # option 2
#    # find false objects here, take 1-step-ahead change in existence probability
#    # and genuity! (which will change b.c. of the new msmt's score)
#    for objidx in range(n_objects):
#        if not associations[0,objidx] or updatepairs[objidx,0]==-1: continue
#        prevobjidx = updatepairs[objidx,0]
#        if objectdetectprobs[prevobjidx] < .1: continue
#        sample = objects[objidx]
#        prevsample = newobjects[prevobjidx]
#        if prevsample[so_ft_pexist] < .05: continue
#        #if sample[so_ft_pexist] < .1: continue
##        matchesgt = False
##        for gtobj in gt:
##            if np.hypot(gtobj['box'][0]-sample[0],gtobj['box'][1]-sample[1])<2.5:
##                matchesgt = True
##        if matchesgt: continue
#        motionreports.append(np.append(prevsample, sample[56:58]))
    
    
#    plotimg1 = plotImgKitti(view_angle)
#    plotimg2 = plotImgKitti(view_angle)
#    # add ground truth
#    for gtobj in gt:
#        box = np.array(gtobj['box'])
#        if gtobj['scored']:
#            addRect2KittiImg(plotimg2, box, (0,0,210*.9,.9))
#        else:
#            addRect2KittiImg(plotimg2, box, (30*.9,80*.9,255*.9,.9))
#    # add measurements
#    for msmt in cammsmts:
#        addRect2KittiImg(plotimg1, msmt[:5], (45,255.9*.5,45,.5))
#    for msmt in allboxes:
#        plotRectangleEdges(plotimg1, msmt[:5], (50,255.9*.8,50,.8))
#    if len(allfrags) > 0:
#        fragpltshape = ((0,0),(0,1),(1,0),(0,-1),(-1,0))
#        fragsplot = (np.array(allfrags) * (-320/30.) + (640,320)).astype(int)
#        fragsplot = fragsplot[(fragsplot[:,0]>0) & (fragsplot[:,0]<639)
#                              & (fragsplot[:,1]>0) & (fragsplot[:,1]<639)]
#        plotPoints(plotimg1, fragsplot[:,0], fragsplot[:,1], fragpltshape, 
#                   (50,150,50,.9))
#    # shade each tile by occupancy
#    for tilex, tiley in np.ndindex(*grndlen):
#        tilecenterx = (tilex+grndstart[0] + .5)*grndstep[0]
#        tilecentery = (tiley+grndstart[1] + .5)*grndstep[1]
#        if tilecenterx < 1 or tilecenterx > 58 or abs(tilecentery) > 28:
#            continue
#        color = 255.9*(1-occupancy[tilex, tiley])#visibility[tilex,tiley]#
#        addRect2KittiImg(plotimg2, (tilecenterx, tilecentery, 0., 1.5, 1.5),
#                         np.array((color*.2, color*.2, color*.2, .2)))
#    # add estimates
#    gggg = associations[0] & (objects[:,so_ft_pexist]*objects[:,57] > .5)
#    for objidx in range(n_objects):
#        if not associations[0,objidx]: continue
#        reportscore, obj = soReport(objects[objidx])
#        label = objects[objidx,56+len(lasers2use)+9]
#        color = (label % 32) / 40.
#        if color > .55: color += .2
#        if reportscore > .5:
#            addRect2KittiImg(plotimg2, obj, hsv2Rgba(color,.9,1.,.8))
#        elif reportscore > .1:
#            addRect2KittiImg(plotimg2, obj, hsv2Rgba(color,1.,1.,.15))
##    for report in fullreports:
##        color = hsv2Rgba(report[6]%16/16., .9, 1., .8)
##        addRect2KittiImg(plotimg2, report[:5], color)
#    # put the plot on top of the camera image to view, display
#    plotimg1 = np.minimum((plotimg1[:,:,:3]/plotimg1[:,:,3:]),255.).astype(np.uint8)
#    plotimg2 = np.minimum((plotimg2[:,:,:3]/plotimg2[:,:,3:]),255.).astype(np.uint8)
#    img = img[:368]
#    display_img = np.zeros((640+img.shape[0], 1280, 3), dtype=np.uint8)
#    display_img[:640, :640] = plotimg1
#    display_img[:640, 640:] = plotimg2
#    display_img[640:, (1280-img.shape[1])//2:(1280+img.shape[1])//2] = img
#    imshow('a', display_img);
#    if waitKey(500) == ord('q'):
#        break
    #imwrite('imgs/{:03d}.png'.format(fileidx-90), display_img)
#    if videofile is not None: video.append_data(display_img[:,:,::-1])
#if videofile is not None: video.close()

#destroyWindow('a')