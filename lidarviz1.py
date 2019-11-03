#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last mod 8/17/19
testing pieces of lidar processing code
tests segmentation and rectangle generation
"""

import numpy as np
from imageio import imread
from cv2 import imshow, waitKey, destroyWindow

from segment import lasers2use, segmentPoints, getGrndHeight

from plotStuff import base_image, plotRectangle, plotPoints, drawLine
from plotStuff import plotImgKitti, addRect2KittiImg, hsv2Rgba
from calibs import calib_extrinsics, calib_projections, view_by_day
from trackinginfo import sceneranges
from trackinginfo import calib_map_training as calib_map
from analyzeGT import readGroundTruthFileTracking

lidar_files = '/home/m2/Data/kitti/tracking_velodyne/training/{:04d}/{:06d}.bin'
img_files = '/home/m2/Data/kitti/tracking_image/training/{:04d}/{:06d}.png'
gt_files = '/home/m2/Data/kitti/tracking_gt/{:04d}.txt'
ground_files = '/home/m2/Data/kitti/tracking_ground/training/{:02d}f{:06d}.npy'
scene_idx = 9#1
startfileidx, endfileidx = sceneranges[scene_idx]
startfileidx = 0#368
endfileidx = 3#369

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
    img = (img.astype(float)*.9).astype(np.uint8)
    img[:,:,1:] += 25
    data = np.fromfile(lidar_files.format(scene_idx, fileidx),
                   dtype=np.float32).reshape((-1,4))[:,:3]
    gt = gt_all[fileidx]
    ground = np.load(ground_files.format(scene_idx, fileidx))
    
    
    starts = np.where(np.diff(np.sign(data[:,1])) > 0)[0]
    starts = np.concatenate(([0], starts+1, [len(data)]))
    true_starts = np.append(np.diff(starts) > 2, [True])
    starts = starts[true_starts]
    assert starts.shape[0] > 55
    lidar = []
    for laser in lasers2use[::-1]:
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
        if gtobj['scored']:
            addRect2KittiImg(plotimg1, box, (0,0,210*.9,.5))
        else:
            addRect2KittiImg(plotimg1, box, (30*.9,80*.9,255*.9,.5))

    # draw laser points
    for lidaridx, pts in enumerate(lidar):
        pts2plot = pts.dot(bevTM[:,:3].T) + bevTM[:,3]
        include_scatter = ((pts2plot[:,0] > 0) & (pts2plot[:,0] < 639) &
                           (pts2plot[:,1] > 0) & (pts2plot[:,1] < 639))
        pts2plot = pts2plot[include_scatter].astype(int)
        plotPoints(plotimg1, pts2plot[:,0], pts2plot[:,1], ((0,0),), (0.,0.,0.,1.))
        
        ptsheight = pts.dot(calib_extrinsic[2,:3])+calib_extrinsic[2,3]
        groundptangles = np.arctan2(pts[ptsheight < .1,1], pts[ptsheight < .1,0])
        groundptangles = groundptangles.sort()
        segments = segmentPoints(pts)
        
        for segidx, seg in enumerate(segments):
            segmiddle = np.mean(seg,axis=0)
            segmiddle = calib_extrinsic[:3,:3].dot(segmiddle) + calib_extrinsic[:3,3]
            seggroundelev = getGrndHeight(ground, segmiddle[0], segmiddle[1])
            segmidheight = segmiddle[2] - seggroundelev
            heights = seg.dot(calib_extrinsic[2,:3])+calib_extrinsic[2,3]
            segisground = max(heights)-seggroundelev < .3
            seginclude = (not segisground) and segmiddle[2]-seggroundelev < 2.
            
            if not seginclude:
                color = (0.,0.,0.,.4)
            else:
                color = hsv2Rgba(lidaridx/len(lidar), .8, 1., 1.)
                #color = hsv2Rgba(segidx % 16 / 16., 8., 1., 1.)
            
            seg2plot = seg.dot(bevTM[:,:3].T) + bevTM[:,3]
            include_scatter = ((seg2plot[:,0] > 0) & (seg2plot[:,0] < 639) &
                               (seg2plot[:,1] > 0) & (seg2plot[:,1] < 639))
            seg2plot = seg2plot[include_scatter].astype(int)
            plotPoints(plotimg2, seg2plot[:,0], seg2plot[:,1], ((0,0),), color)
            
#            if len(segment) > 2:
#                rect = makeMeasurement(segment)
#                rec = uv2xy(rect)
#                consider = rec[3] < 3 and rec[4] < 3 and (rec[3]>1 or rec[4]>1)
#                if consider:
#                    color = hsv2Rgba(segidx % 16 / 16., 1., 1., .4)
#                    addRect2KittiImg(plotimg2, rec, color)  
    
    # add lidar points to image
    for laser in lasers2use[::-1]:
        pts = data[starts[laser]:starts[laser+1]]
        pts2plot = pts.dot(calib_projection[:3,:3].T)+calib_projection[:3,3]
        pts2plot[:,:2] /= pts2plot[:,2,None]
        include_scatter = np.all(pts2plot>=0, axis=1)
        include_scatter &= (pts2plot[:,0]<img.shape[0]) & (pts2plot[:,1]<img.shape[1])
        pts2plot = pts2plot[include_scatter,:2].astype(int)
        img[pts2plot[:,0],pts2plot[:,1]] = (255,0,0)
    
    plotimg1 = np.minimum((plotimg1[:,:,:3]/plotimg1[:,:,3:]),255.).astype(np.uint8)
    plotimg2 = np.minimum((plotimg2[:,:,:3]/plotimg2[:,:,3:]),255.).astype(np.uint8)
    img = img[:368]
    display_img = np.zeros((640+img.shape[0], 1280, 3), dtype=np.uint8)
    display_img[:640, :640] = plotimg1
    display_img[:640, 640:] = plotimg2
    display_img[640:, (1280-img.shape[1])//2:(1280+img.shape[1])//2] = img
    imshow('a', display_img);
    if waitKey(400) == ord('q'):
        break