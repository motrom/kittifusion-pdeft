#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last mod 8/20/19
testing pieces of lidar processing code
this piece includes the occlusion map, measurement shape bounding,
and measurement selection (using shape criterion)
"""

import numpy as np
from imageio import imread
from cv2 import imshow, waitKey, destroyWindow

from segment import lasers2use, segmentPoints, makeMeasurement, uv2xy, laser_angles
from segment import getGroundForLaser, makeOcclusionMap, boundMeasurement
from segment import getGrndHeight, groundforlaser_angles

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
scene_idx = 1
startfileidx, endfileidx = sceneranges[scene_idx]
startfileidx = 182
endfileidx = 183
occlusionlaser = 1

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

#    # get angles of all ground points
#    # this is cheating and using all 64 lasers atm
#    # but making it use only available lasers requires two separate loops...
#    heights = data.dot(calib_extrinsic[2,:3]) + calib_extrinsic[2,3]
#    groundpts = (heights < .1) & (data[:,0] > .1)
#    groundptangles = np.arctan2(data[groundpts,1], data[groundpts,0])
#    groundptangles.sort()    
    
    starts = np.where(np.diff(np.sign(data[:,1])) > 0)[0]
    starts = np.concatenate(([0], starts+1, [len(data)]))
    true_starts = np.append(np.diff(starts) > 2, [True])
    starts = starts[true_starts]
    assert starts.shape[0] > 55
    lidar = []
    for laser in [lasers2use[occlusionlaser]]:
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
        addRect2KittiImg(plotimg1, box, (220.,220.,220,.5))

    # draw laser points
    for laseridx, pts in enumerate(lidar):
        segs = segmentPoints(pts)
        # get rectangles from point segments
        msmts = [makeMeasurement(seg) for seg in segs]
        segsareground = []
        segsinclude = []
        
        for segidx, seg in enumerate(segs):
            segmiddle = np.mean(seg,axis=0)
            segmiddle = calib_extrinsic[:3,:3].dot(segmiddle) + calib_extrinsic[:3,3]
            seggroundelev = getGrndHeight(ground, segmiddle[0], segmiddle[1])
            segmidheight = segmiddle[2] - seggroundelev
            heights = seg.dot(calib_extrinsic[2,:3])+calib_extrinsic[2,3]
            segisground = max(heights)-seggroundelev < .3
            seginclude = (not segisground) and segmiddle[2]-seggroundelev < 2.
            segsareground.append(segisground)
            segsinclude.append(seginclude)
            
            if seginclude:
                rect = msmts[segidx]
                rec = np.array(uv2xy(rect))
                consider = rec[3] < 3 and rec[4] < 3 and (rec[3]>.5 or rec[4]>.5)
                if consider:
                    rec[:2] = calib_extrinsic[:2,:2].dot(rec[:2]) +\
                                calib_extrinsic[:2,2]
                    color = hsv2Rgba(segidx % 16 / 16., 1., 1., .4)
                    plotRectangleEdges(plotimg1, rec, color)
            
            color = (0.,0.,0.,.4)
            seg2plot = seg.dot(bevTM[:,:3].T) + bevTM[:,3]
            include_scatter = ((seg2plot[:,0] > 0) & (seg2plot[:,0] < 639) &
                               (seg2plot[:,1] > 0) & (seg2plot[:,1] < 639))
            seg2plot = seg2plot[include_scatter].astype(int)
            plotPoints(plotimg1, seg2plot[:,0], seg2plot[:,1], ((0,0),), color)
            
        ground_present = getGroundForLaser(ground, laser_angles[laser])
        starting_angle = min(-view, np.arctan2(pts[0,1],pts[0,0]))
        ending_angle = max(view, np.arctan2(pts[-1,1],pts[-1,0]))
        occlusion_map = makeOcclusionMap(segs, segsareground, msmts,
                                         starting_angle, ending_angle, True,
                                         #ground_present, groundptangles)
                                         ground_present, None)
        color = (239,239,239,1.)
        drawOcclusion(plotimg2, occlusion_map, color)
        
        # the earlier occlusion secshun
        detectzone = getGroundForLaser(ground, laser_angles[laser],
                                   laser_height=1.65-1.3)
        detectzonecanvas = np.zeros(plotimg2.shape)
        detectzoneasoccmap = np.array((groundforlaser_angles,
                                              detectzone, detectzone)).T
        drawOcclusion(detectzonecanvas, detectzoneasoccmap, (1,1,1,1.))
        plotimg2[detectzonecanvas[:,:,0]==0] = (244,244,244,1.)
        
        outputmsmts = []
        for segidx in range(len(segs)):
            if not segsinclude[segidx]:
                continue
            msmt = msmts[segidx]
            msmt = boundMeasurement(segidx, msmt, occlusion_map)
            color = hsv2Rgba(segidx % 16 / 16., 1., 1., .75)
            rec = np.array(uv2xy(msmt[:6]))
            rec[:2] = calib_extrinsic[:2,:2].dot(rec[:2]) +\
                                calib_extrinsic[:2,2]
            addRect2KittiImg(plotimg2, rec, color)
            minlen = msmt[3]-msmt[2]
            minwid = msmt[5]-msmt[4]
            maxlen = msmt[7]-msmt[6]
            maxwid = msmt[9]-msmt[8]
            # check whether msmt could be car shape
            ismain = minlen<5.5 and minwid<2.2 and maxlen>2.95 and maxwid>1.35
            ismain |= minwid<5.5 and minlen<2.2 and maxwid>2.95 and maxlen>1.35
            ismain &= minlen > 1. or minwid > 1. # observed a good line segment
            if ismain:
                outputmsmts.append(msmt)
                rec = np.array(uv2xy(msmt[:2]+msmt[6:]))
                rec[:2] += calib_extrinsic[:2,2]
                
                plotRectangleEdges(plotimg2, rec, color)
            
    
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