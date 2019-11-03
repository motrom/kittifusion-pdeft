#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 6/4/19
"""

import numpy as np
from scipy.optimize import linear_sum_assignment 
import matplotlib.pyplot as plt


overlapres = 50
overlapbox = np.mgrid[:float(overlapres), :float(overlapres)]
overlapbox += .5
overlapbox *= 2./overlapres
overlapbox -= 1
overlapbox = overlapbox.transpose((1,2,0))
def soMetricIoU(boxa, boxb):
    relx = boxa[0]-boxb[0]
    rely = boxa[1]-boxb[1]
    ca, sa = np.cos(boxa[2]), np.sin(boxa[2])
    cb, sb = np.cos(boxb[2]), np.sin(boxb[2])
    la,wa = boxa[3:5]
    lb,wb = boxb[3:5]
    R = np.array([[la/lb*(ca*cb+sa*sb), wa/lb*(ca*sb-cb*sa)],
                  [la/wb*(cb*sa-ca*sb), wa/wb*(ca*cb+sa*sb)]])
    t = np.array([(cb*relx + sb*rely)/lb, (cb*rely - sb*relx)/wb])
    grid = np.einsum(R, [0,1], overlapbox, [2,3,1], [2,3,0]) + t
    intersection = np.sum(np.all(abs(grid) < 1, axis=2))
    ioa = float(intersection) / overlapres**2
    return ioa / (1 - ioa + lb*wb/la/wa)

def soMetricEuc(boxa, boxb):
    eucdist = np.hypot(boxa[0]-boxb[0],boxa[1]-boxb[1])
    angledist = abs((boxa[2]-boxb[2] + np.pi/2)%np.pi - np.pi/2)
    return eucdist + angledist/2 < 3.

    
class MetricMine2():
    def __init__(self):
        self.dets = []
        self.switchscores = []
        self.nmissed = 0
        self.previousids = {}
        self.previousscores = {}
        self.newscene = True
    def newScene(self):
        self.previousids = {}
        self.previousscores = {}
    def okMetric(self, boxa, boxb):
        return soMetricIoU(boxa, boxb) > .3
    def goodMetric(self, boxa, boxb):
        return soMetricIoU(boxa, boxb) > .7
    def add(self, gt, gtscored, gtdifficulty, gtids, ests, scores, estids):
        ngt = gt.shape[0]
        assert gtscored.shape[0] == ngt
        assert gtdifficulty.shape[0] == ngt
        nests = ests.shape[0]
        assert scores.shape[0] == nests
        gtscored = gtscored & (gtdifficulty < 3)
        estorder = np.argsort(scores)[::-1]
        gtopen = np.ones(ngt, dtype=bool)
        currentids = {}
        currentscores = {}
        for estidx in estorder:
            bestgtGood = False
            bestgtScored = False
            bestgtidx = None
            for gtidx in range(ngt):
                if gtopen[gtidx] and self.okMetric(gt[gtidx], ests[estidx]):
                    keep = False
                    swap = bestgtidx is None
                    goodfit = self.goodMetric(gt[gtidx], ests[estidx])
                    isscored = gtscored[gtidx]
                    if not swap:
                        keep = bestgtGood and not goodfit
                        swap = bestgtGood and goodfit
                    if not keep and not swap:
                        swap = not bestgtScored and isscored
                    if swap:
                        bestgtidx = gtidx
                        bestgtGood = goodfit
                        bestgtScored = isscored
            if bestgtidx is None:
                self.dets.append((scores[estidx], False, False))
            else:
                gtopen[bestgtidx] = False
            if bestgtScored:
                self.dets.append((scores[estidx], True, bestgtGood))
                # search for id swap
                gtid = gtids[bestgtidx]
                switch = (gtid in self.previousids and
                          self.previousids[gtid] != estids[estidx])
                if switch:
                    switchscore = min(self.previousscores[gtid], scores[estidx])
                    self.switchscores.append(switchscore)
                currentids[gtid] = estids[estidx]
                currentscores[gtid] = scores[estidx]
        self.nmissed += sum(gtopen & gtscored)
        self.previousids = currentids
        self.previousscores = currentscores
    def calc(self):
        dets = np.array(sorted(self.dets)[::-1])
        switchscores = -np.array(sorted(self.switchscores)[::-1])
        ndets = len(dets)
        nt = sum(dets[:,1]) + self.nmissed
        tps = np.cumsum(dets[:,1])
        checkpts = np.append(np.where(np.diff(dets[:,0]))[0], ndets-1)
        rec = tps[checkpts] / nt
        prec = tps[checkpts] / (checkpts+1)
        goodtpr = (np.cumsum(dets[:,2]))[checkpts] / nt
        switches = np.searchsorted(switchscores, -dets[checkpts,0])
        #mota = (2*tps[checkpts] - checkpts-1 - switches) / float(nt)
        rec = np.concatenate(([0.], rec, [rec[-1]]))
        prec = np.concatenate(([1.], prec, [0.]))
        goodtpr = np.concatenate(([0.], goodtpr, [goodtpr[-1]]))
        switches = np.concatenate(([switches[0]], switches, [switches[-1]]))
        return np.array((rec, prec, goodtpr, switches)).T
    def calcMOTA(self):
        dets = np.array(sorted(self.dets)[::-1])
        switchscores = -np.array(sorted(self.switchscores)[::-1])
        ndets = len(dets)
        nt = sum(dets[:,1]) + self.nmissed
        tps = np.cumsum(dets[:,1])
        checkpts = np.append(np.where(np.diff(dets[:,0]))[0], ndets-1)
        switches = np.searchsorted(switchscores, -dets[checkpts,0])
        mota = (2*tps[checkpts] - checkpts-1 - switches) / float(nt)
        return max(mota)
    
        
" counts = 210, 1063, 369, 268, 679, 854, 378, 1843, 522, 2121"
""

if __name__ == '__main__':
    """
        runs a single accuracy metric across multiple scenes
        formatForKittiScore gets rid of things kitti didn't annotate
    """
    from calibs import calib_extrinsics, calib_projections, view_by_day
    from trackinginfo import sceneranges
    from trackinginfo import calib_map_training as calib_map
    from analyzeGT import readGroundTruthFileTracking, formatForKittiScoreTracking
    from imageio import imread
    
    scenes = [0,1,2,3,4,5,6,7,8,9]
    nframesahead = 0
    tests = [('trMGL3', 'MGL', 'b'),
             ('trMGLnofake3', 'w/o genuity', 'g--'),
             ('trackingresultsMGR', 'camera', 'k-.'),
             ('trMGLnodet3', 'w/o detectability', 'r:')]

    
    
    gt_files = '/home/m2/Data/kitti/tracking_gt/{:04d}.txt'
    estfiles = '/home/m2/Data/kitti/estimates/{:s}/{:02d}f{:04d}.npy'
    img_files = '/home/m2/Data/kitti/tracking_image/training/{:04d}/000000.png'
    ground_plane_files = '/home/m2/Data/kitti/tracking_ground/training/{:02d}f{:06d}.npy'
    
    results = []
    motas = []
    
    for testfolder, testname, testcolor in tests:
        metric = MetricMine2()
        
        for scene_idx in scenes:
            # run some performance metrics on numpy-stored results
            startfile, endfile = sceneranges[scene_idx]
            #startfile = 200
            #endfile = 40
            startfile += nframesahead
            calib_idx = calib_map[scene_idx]
            calib_extrinsic = calib_extrinsics[calib_idx].copy()
            calib_extrinsic[2,3] += 1.65
            view_angle = view_by_day[calib_idx]
            calib_projection = calib_projections[calib_idx]
            calib_projection = calib_projection.dot(np.linalg.inv(calib_extrinsic))
            imgshape = imread(img_files.format(scene_idx)).shape[:2]
            with open(gt_files.format(scene_idx), 'r') as fd: gtfilestr = fd.read()
            gt_all, gtdontcares = readGroundTruthFileTracking(gtfilestr,('Car','Van'))
            metric.newScene()
            
            for fileidx in range(startfile, endfile):
                ground = np.load(ground_plane_files.format(scene_idx, fileidx))
                
                ests = np.load(estfiles.format(testfolder, scene_idx, fileidx))
                estids = ests[:,6].astype(int)
                scores = ests[:,5]
                ests = ests[:,:5]
                rede = formatForKittiScoreTracking(ests, estids, scores, fileidx,
                                    ground, calib_projection, imgshape, gtdontcares)
                ests = np.array([redd[0] for redd in rede])
                scores = np.array([redd[2] for redd in rede])
                estids = np.array([redd[1] for redd in rede])
                
                gthere = gt_all[fileidx]
                gtboxes = np.array([gtobj['box'] for gtobj in gthere])
                gtscores = np.array([gtobj['scored'] for gtobj in gthere],dtype=bool)
                gtdifficulty = np.array([gtobj['difficulty'] for gtobj in gthere],
                                        dtype=int)
                gtids = np.array([gtobj['id'] for gtobj in gthere],dtype=int)
                gtdontcareshere = gtdontcares[fileidx]
                
                metric.add(gtboxes, gtscores, gtdifficulty, gtids,
                           ests, scores, estids)
        restest = metric.calc()
        results.append((testname, restest, testcolor))
        motas.append(metric.calcMOTA())
        
        
    # nodet is currently nofakelogic
    fig, axeses = plt.subplots(1, 3, figsize=(12., 3.))
    plt1, plt2, plt3 = axeses.flat
    plt1.set_xlim((0.5, 1.))
    plt2.set_xlim((0.5, 1.))
    plt3.set_xlim((0.5, 1.))
    plt1.set_ylim((0.5, 1.))
    plt2.set_ylim((0., 1.))
    plt1.set_title('Precision vs Recall')
    plt2.set_title('Close fit recall vs Recall')
    plt3.set_title('# identity swaps vs Recall')
    maxswaps = int(max(np.max(result[1][:,3]) for result in results))+1
    plt3.set_yticks(list(range(0, maxswaps, maxswaps//5+1)))
    for testname, result, color in results:
        plt1.plot(result[:,0], result[:,1], color, label=testname)
        plt2.plot(result[:,0], result[:,2], color, label=testname)
        plt3.plot(result[:,0], result[:,3], color, label=testname)
    #plt3.legend(loc='center right')
    plt3.legend(bbox_to_anchor = (1.04, 1), loc="upper left")
    #plt1.legend(bbox_to_anchor = (0., -0.05), loc="upper left", ncol=4)
    plt.show()