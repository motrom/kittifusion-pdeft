#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last mod 8/25/19
testing pieces of lidar processing code
this piece includes separation of measurements into full-car and part-of-car
and conversion into xyalw format with normal distribution
also includes monogrnet for comparison
"""

import numpy as np
from imageio import imread
from cv2 import imshow, waitKey, destroyWindow

from segment import lasers2use, segmentPoints, makeMeasurement, uv2xy, laser_angles
from segment import getGroundForLaser, makeOcclusionMap, boundMeasurement
from segment import getGrndHeight, msmtBound2msmtNormal

from presavedSensor import getMsmts as getCamMsmts

from plotStuff import base_image, plotRectangle, plotPoints, drawLine
from plotStuff import plotImgKitti, addRect2KittiImg, hsv2Rgba, plotRectangleEdges
from plotStuff import drawOcclusion#fillArc
from calibs import calib_extrinsics, calib_projections, view_by_day
from trackinginfo import sceneranges
from trackinginfo import calib_map_training as calib_map
from analyzeGT import readGroundTruthFileTracking

lidar_files = '/home/m2/Data/kitti/tracking_velodyne/training/{:04d}/{:06d}.bin'
img_files = '/home/m2/Data/kitti/tracking_image/training/{:04d}/{:06d}.png'
gt_files = '/home/m2/Data/kitti/tracking_gt/{:04d}.txt'
ground_files = '/home/m2/Data/kitti/tracking_ground/training/{:02d}f{:06d}.npy'
scene_idx = 6
startfileidx, endfileidx = sceneranges[scene_idx]
startfileidx = 90#135
endfileidx = 150#136
    

def clear(): destroyWindow('a')

calib_idx = calib_map[scene_idx]
calib_extrinsic = calib_extrinsics[calib_idx].copy()
calib_projection = calib_projections[calib_idx]
calib_intrinsic = calib_projection.dot(np.linalg.inv(calib_extrinsic))
calib_extrinsic[2,3] += 1.65
view_angle = view_by_day[calib_idx]
view = view_angle + .1
bevTM = np.array(((-320./30,0,640), (0,-320./30,320)))
bevTM = bevTM.dot(calib_extrinsic[[0,1,3],:]) # lidar xyz to pixels

with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
gt_all, gtdc = readGroundTruthFileTracking(gtfilestr, ('Car', 'Van'))


for fileidx in range(startfileidx, endfileidx):
    img = imread(img_files.format(scene_idx, fileidx))[:,:,::-1]
    data = np.fromfile(lidar_files.format(scene_idx, fileidx),
                   dtype=np.float32).reshape((-1,4))[:,:3]
    gt = gt_all[fileidx]
    ground = np.load(ground_files.format(scene_idx, fileidx)) 
    
    starts = np.where(np.diff(np.sign(data[:,1])) > 0)[0]
    starts = np.concatenate(([0], starts+1, [len(data)]))
    true_starts = np.append(np.diff(starts) > 2, [True])
    starts = starts[true_starts]
    assert starts.shape[0] > lasers2use[-1]+1
    lidar = []
    for laser in lasers2use:
        pts = data[starts[laser]:starts[laser+1]]
        include = pts[:,0] > 0
        include &= abs(pts[:,1]) < pts[:,0]*view + 2.
        include &= pts.dot(calib_extrinsic[2,:3])+calib_extrinsic[2,3] > -.3
        pts = pts[include]
        swap_idx = np.where(np.diff(np.arctan2(pts[:,1],pts[:,0]))<-.05)[0]
        assert len(swap_idx) <= 1
        if len(swap_idx) == 1:
            swap_idx = swap_idx[0] + 1
            pts = np.append(pts[swap_idx:], pts[:swap_idx], axis=0)
        lidar.append(pts.copy())
    
    
    plotimg1 = plotImgKitti(view_angle)
    plotimg2 = plotImgKitti(view_angle)
    # draw object
    for gtobj in gt:
        box = np.array(gtobj['box'])
        addRect2KittiImg(plotimg1, box, (200.,200.,200,.9))
        addRect2KittiImg(plotimg2, box, (200.,200.,200,.9))
        
    cammsmts = getCamMsmts(scene_idx, fileidx)
    for msmt in cammsmts:
        if msmt[5] > .05:
            # high scored msmts are blue, low are purple
            plotRectangleEdges(plotimg2, msmt[:5], hsv2Rgba((1-msmt[5])*.9/.95, 1,1,.8))

    # draw laser points
    for laseridx, pts in enumerate(lidar):
        lasercolor = hsv2Rgba(laseridx / (len(lidar)+2.), 1., 1., .75)
        segs = segmentPoints(pts)
        # get rectangles from point segments
        
        msmts = []
        segsareground = []
        segsinclude = []
        for segidx, seg in enumerate(segs):
            segmiddle = np.mean(seg,axis=0)
            segmiddle = calib_extrinsic[:3,:3].dot(segmiddle) + calib_extrinsic[:3,3]
            seggroundelev = getGrndHeight(ground, segmiddle[0], segmiddle[1])
            heights = seg.dot(calib_extrinsic[2,:3])+calib_extrinsic[2,3]
            segisground = max(heights)-seggroundelev < .3
            seginclude = (not segisground) and segmiddle[2]-seggroundelev < 2.
            segsareground.append(segisground)
            segsinclude.append(seginclude)
            msmts.append(makeMeasurement(seg))
            
        ground_present = getGroundForLaser(ground, laser_angles[laser])
        starting_angle = min(-view, np.arctan2(pts[0,1],pts[0,0]))
        ending_angle = max(view, np.arctan2(pts[-1,1],pts[-1,0]))
        occlusionmap = makeOcclusionMap(segs, segsareground, msmts,
                                        starting_angle, ending_angle, True,
                                        ground_present, None)
        
        outputmsmts = []
        for segidx in range(len(segs)):
            if not segsinclude[segidx]:
                continue
            msmt = msmts[segidx]
            msmt = boundMeasurement(segidx, msmt, occlusionmap)
            if msmt[3]-msmt[2] < 1. and msmt[5]-msmt[4] < 1.:
                msmt1,msmt2 = None,None
            else:
                msmt1, msmt2 = msmtBound2msmtNormal(msmt)
                
            if msmt1 is None and msmt2 is None:
                rec = uv2xy(msmt[:6])[:2]
                rec = calib_extrinsic[:2,:2].dot(rec) + calib_extrinsic[:2,2]
                nottoobig = ((msmt[3]-msmt[2] < 5.88) and
                             (msmt[5]-msmt[4] < 2.28)) or (
                             (msmt[3]-msmt[2] < 2.28) and
                             (msmt[5]-msmt[4] < 5.88))
                nottoosmall = max(msmt[5]-msmt[4],msmt[3]-msmt[2]) > .35
                if not nottoobig:
                    print("toobig dude at {:.1f}  {:.1f}".format(rec[0],rec[1]))
                    #continue #ABCDEEEE
                #if not (nottoobig and nottoosmall): continue
                if not nottoobig: continue
                if nottoosmall: lasercolor = hsv2Rgba(.333, 1., 1., 1.)
                else: lasercolor = hsv2Rgba(0., 1., 1., 1.)    
                if rec[0] < .2 or rec[0] > 59 or abs(rec[1]) > 29.5: continue
                rec = bevTM.dot((rec[0],rec[1],0,1)).astype(int)
                scattershape = np.mgrid[-2:3,-2:3].transpose((1,2,0)).reshape((25,2))
                scattershape = scattershape[np.hypot(scattershape[:,0],
                                                     scattershape[:,1])<2.25]
                assignlasercolor = np.append(lasercolor[:3]/lasercolor[3], (1.,))
                plotPoints(plotimg1, rec[None,0], rec[None,1], scattershape,
                           assignlasercolor)
                
            if msmt1 is not None:
                rec, cov = msmt1
                assert cov[0,0] < 1000 and cov[1,1] < 1000
                rec[:2] = calib_extrinsic[:2,:2].dot(rec[:2]) +\
                                    calib_extrinsic[:2,2]
                vardiff = (cov[0,0]-cov[1,1])/2
                varmean = (cov[0,0]+cov[1,1])/2
                stdinmaindirection = (varmean + np.hypot(vardiff, cov[0,1]))**.5
                color = hsv2Rgba(min(stdinmaindirection/4.5,.9), 1., 1., 1.)
                plotRectangleEdges(plotimg1, rec, color)
                outputmsmts.append(msmt1)
            if msmt2 is not None:
                rec, cov = msmt2
                assert cov[0,0] < 1000 and cov[1,1] < 1000
                rec[:2] = calib_extrinsic[:2,:2].dot(rec[:2]) +\
                                    calib_extrinsic[:2,2]
                vardiff = (cov[0,0]-cov[1,1])/2
                varmean = (cov[0,0]+cov[1,1])/2
                stdinmaindirection = (varmean + np.hypot(vardiff, cov[0,1]))**.5
                color = hsv2Rgba(min(stdinmaindirection/4.5,.9), 1., 1., 1.)
                plotRectangleEdges(plotimg1, rec, color)
                outputmsmts.append(msmt2)
                
    
    plotimg1 = np.minimum((plotimg1[:,:,:3]/plotimg1[:,:,3:]),255.).astype(np.uint8)
    plotimg2 = np.minimum((plotimg2[:,:,:3]/plotimg2[:,:,3:]),255.).astype(np.uint8)
    img = img[:368]
    display_img = np.zeros((640+img.shape[0], 1280, 3), dtype=np.uint8)
    display_img[:640, :640] = plotimg1
    display_img[:640, 640:] = plotimg2
    display_img[640:, (1280-img.shape[1])//2:(1280+img.shape[1])//2] = img
    imshow('a', display_img);
    if waitKey(500) == ord('q'):
        break