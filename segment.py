#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:59:18 2019

@author: m2
"""
import numpy as np
from math import hypot, atan2
from config import grndstep, grndstart, grndlen
from rectools import uv2xy, fitRecToMold, uvBound2xyNormal

series_distance_cutoff = .45 ** 2 # m
min_points_in_segment = 3
concave_split_cutoff = 1. # m
def segmentPoints(points):
    if len(points) < 3: return []
    diff = np.diff(points[:,:2], axis=0)
    distsq = diff[:,0]*diff[:,0] + diff[:,1]*diff[:,1]
    cuts = np.where(distsq > series_distance_cutoff)[0]
    if cuts.shape[0]==0:
        return [points] # all contiguous
    # can't think of a decent numpy way to do this next part
    fixed_cuts = [0]
    outlier_idxs = [[]]
    if cuts[0] == 0: # first point is outlier
        fixed_cuts.append(1)
        outlier_idxs.append([])
        cuts = cuts[1:]
    for cut in cuts:
        if cut == fixed_cuts[-1]:
            # this point was an outlier
            outlier_skipped = points[cut+1,:2]-points[cut-1,:2]
            outlier_skipped_distance = outlier_skipped[0]*outlier_skipped[0] +\
                                        outlier_skipped[1]*outlier_skipped[1]
            if outlier_skipped_distance <= series_distance_cutoff:
                fixed_cuts.pop() # the cut was only due to this outlier
                outlier_idxs.pop()
            outlier_idxs[-1].append(cut - fixed_cuts[-1])
        else:
            fixed_cuts.append(cut+1)
            outlier_idxs.append([])
    if fixed_cuts[-1]==len(points)-1: # last element is an outlier
        outlier_idxs.pop() # just leave it out of the cuts
    else:
        fixed_cuts.append(len(points))
    n_cuts = len(fixed_cuts) - 1
    
    segs = [np.delete(points[fixed_cuts[idx]:fixed_cuts[idx+1]], outlier_idxs[idx], axis=0)
            for idx in range(n_cuts)]
    
    # remove segments that are too small
    # also, check for high concavity - possibly inward corner
    fixed_segs = []
    for seg in segs:
        if seg.shape[0] < min_points_in_segment:
            continue
        seg_beginning = (seg[0]+seg[1])/2
        seg_end = (seg[-2]+seg[-1])/2
        u = seg_end[1]-seg_beginning[1] # ux+vy=c
        v = seg_beginning[0]-seg_end[0]
        c = seg_end[1]*seg_beginning[0]-seg_end[0]*seg_beginning[1]
        uvlen = hypot(u,v)
        points_distconcave = u*seg[:,0]+v*seg[:,1]-c
        split_point = np.argmax(points_distconcave)
        if points_distconcave[split_point] > concave_split_cutoff * uvlen:
            #print("splitting on concavity {:.1f}".format(points_distconcave[split_point]))
            if split_point >= min_points_in_segment:
                fixed_segs.append(seg[:split_point])
            if split_point <= seg.shape[0]-min_points_in_segment:
                fixed_segs.append(seg[split_point:])
        else:
            fixed_segs.append(seg)
    return fixed_segs

min_edge_len = .05

def makeMeasurement(pts):
    if pts.shape[0] < 2:
        raise Exception("can't segment fewer than 2 points")
    elif pts.shape[0] == 2:
        u = pts[1][0] - pts[0][0]
        v = pts[1][1] - pts[0][1]
        uvlen = hypot(u,v)
        u /= uvlen
        v /= uvlen
        ulo = pts[0][0]*u + pts[0][1]*v
        vlo = pts[0][0]*v - pts[0][1]*u
        uvdiff = max((min_edge_len - uvlen)*.5, 0) # enforce minimum edge len
        return (u,v,ulo-uvdiff,ulo+uvlen+uvdiff,vlo,vlo+min_edge_len)
    else:
        rec = marShrinkSearch(pts)
    # keep a minimum size for objects
    rec = (rec[0], rec[1], rec[2], max(rec[2]+min_edge_len, rec[3]),
           rec[4], max(rec[4]+min_edge_len, rec[5]))
    return rec#standardize(rec) # should always be standardized


halfpi = np.pi/2
shrinksearch_initres = halfpi / 5
shrinksearch_initangles = np.arange(0, halfpi, shrinksearch_initres)
shrinksearch_finalres = .03 # stop once resolution below this is used
def marShrinkSearch(pts):
    angle_limit = np.arctan2(pts[-1,1], pts[-1,0])
    res = shrinksearch_initres
    checks = shrinksearch_initangles + angle_limit
    best_fit = 1e6
    best_angle = None
    best_rec = None
    while res > shrinksearch_finalres:
        for angle in checks:
            c,s = np.cos(angle), np.sin(angle)
            u = c*pts[:,0] + s*pts[:,1]
            v = s*pts[:,0] - c*pts[:,1]
            uhi = max(u)
            vhi = max(v)
            ulo = min(u)
            vlo = min(v)
            fit = scoreVisibleArea(u,v, ulo,uhi,vlo,vhi)
            if fit < best_fit:
                best_fit = fit
                best_rec = (c, s, ulo,uhi,vlo,vhi)
                best_angle = angle
        res *= .5
        checks = ((best_angle+res-angle_limit)%halfpi + angle_limit, 
                  (best_angle-res-angle_limit)%halfpi + angle_limit)
    return best_rec

def scoreArea(u, v, ulo,uhi,vlo,vhi):
    return (uhi-ulo)*(vhi-vlo)

def scoreSumDist(u, v, ulo,uhi,vlo,vhi):
    udist = np.minimum(u-ulo, uhi-u)
    vdist = np.minimum(v-vlo, vhi-v)
    return sum(np.minimum(udist, vdist))

def scoreVisibleDist(u, v, ulo,uhi,vlo,vhi):
    if ulo < 0:
        return sum(v-vlo)
    return sum(np.minimum(u-ulo, v-vlo))

def scoreVisibleArea(u,v, ulo,uhi,vlo,vhi):
    v = v - vlo
    vhi -= vlo
    area = u[1:].dot(v[:-1])
    area -= u[:-1].dot(v[1:])
    area += u[-1]*v[-1]
    area -= u[0]*v[0]
    if ulo < 0:
        area += (u[0]-ulo)*v[0]*2
    else:
        area += (u[0]-ulo)*vhi*2
    area += (uhi-u[-1])*v[-1]*2
    return area


""" assuming laser forms a cone directly downward (which is close)
    finds distance in each angle
    as of 8/18/19 returns distance instead of boolean
"""
groundforlaser_angles = np.arange(-1.35, 1.4, .1)
lowestdistance = 4.
def getGroundForLaser(ground, laser_tan, laser_height=1.65, max_distance = 50.):
    ground_present = np.zeros(groundforlaser_angles.shape)
    tilexmin = grndstart[0]*grndstep[0]
    tilexmax = tilexmin + grndlen[0]*grndstep[0]
    tileymin = grndstart[1]*grndstep[1]
    tileymax = tileymin + grndlen[1]*grndstep[1]
    for angle_idx, angle in enumerate(groundforlaser_angles):
        ax = np.cos(angle)
        ay = np.sin(angle)
        x,y = 0,0
        distance = lowestdistance
        x,y = ax*distance, ay*distance
        grounddistance = 1e3
        while x>tilexmin and x<tilexmax and y>tileymin and y<tileymax:
            z = laser_height + laser_tan*distance
            tilex = int(x/grndstep[0])-grndstart[0]
            tiley = int(y/grndstep[1])-grndstart[1]
            height = ground[tilex, tiley].dot((x,y,z,-1))
            if height <= 0 or height > 2.1:
                grounddistance = distance
                break
            distance += 2.
            x,y = ax*distance, ay*distance
        ground_present[angle_idx] = grounddistance
    return ground_present
                

inf = 1e3 # meters, suitably large number
angle_acceptable_overlap = .03 # radians
occlusion_resolution = .01 # radians
def makeOcclusionMap(segs, segsareground, recs, starting_angle, ending_angle,
                      hard_ends = True, ground_present=None, ground_points=None):
    """
inputs:
    segs = segmented points (only first and last are used, except for ground segs)
    seg_ground = boolean list of whether each segment is ground
    recs = rectangle approximation of each segment, standardized
    starting_angle = in radians, beginning of visible range
    ending_angle = in radians, end of visible range
    hard_ends = whether msmts extending beyond starting or ending angles
                are considered occluded or not
    ground_present
        if None: absorption is ignored (treated like absence of object)
        else: boolean array
            stating whether each angle of this laser hits ground or not
output:
    Nx5 array, columns = (angle, distance1, distance2, cw segment idx, cc segment idx)
    objects occluded the line segments between each row
    for instance if row 5 = (theta1, d1, d2) and row6 = (theta2, d3, d4)
    there is an occluding line segment between cos(theta1)*d2, sin(theta1)*d2
                and cos(theta2)*d3, sin(theta2)*d3
    """
    if len(segs) == 0: return np.zeros((0,5))
    
    occlusion_map = np.zeros((len(segs)*3+2, 5))
    end_distances = 0. if hard_ends else inf
    #starting_angle = -1.35
    #ending_angle = 1.35
    occlusion_map[0] = (starting_angle, end_distances, inf, -1, -1)
    occlusion_map[-1] = (ending_angle, inf, end_distances, len(segs), len(segs))
    
    for seg_idx in range(len(segs)):
        seg = segs[seg_idx]
        seg_is_ground = segsareground[seg_idx]
        rec = recs[seg_idx]
        first_point = seg[0]
        last_point = seg[-1]
        first_angle = atan2(first_point[1], first_point[0])
        last_angle = atan2(last_point[1], last_point[0])
        if seg_is_ground:
            # for concave ground segments, use farther point for occlusion distance
            median_point = seg[len(seg) // 2]
            first_dist = hypot(median_point[0], median_point[1])
            last_dist = first_dist
            corner_angle = atan2(median_point[1], median_point[0])
            corner_dist = first_dist
        else:
            # for convex object detections, use endpoints
            # will also use midpoint taken from rectangle fit, later
            first_dist = hypot(first_point[0], first_point[1])
            last_dist = hypot(last_point[0], last_point[1])
            if rec[2] > 0:
                # two sides visible
                corner_x = rec[0]*rec[2] + rec[1]*rec[4] # major error fixed 8/20
                corner_y = rec[1]*rec[2] - rec[0]*rec[4]
                corner_angle = atan2(corner_y, corner_x)
                if (corner_angle > first_angle + occlusion_resolution and
                             last_angle > corner_angle + occlusion_resolution):
                    corner_dist = hypot(corner_y, corner_x)
                else:
                    corner_angle = first_angle
                    corner_dist = first_dist
            else:
                # flat object
                corner_angle = first_angle
                corner_dist = first_dist
        occlusion_map[seg_idx*3+1] = (first_angle, inf, first_dist, seg_idx, seg_idx)
        occlusion_map[seg_idx*3+2] = (corner_angle, corner_dist, corner_dist,
                                         seg_idx, seg_idx)
        occlusion_map[seg_idx*3+3] = (last_angle, last_dist, inf, seg_idx, seg_idx)
        
    # prune and search for errors    
    new_map = []
    point = tuple(occlusion_map[0])
    for nextpoint in occlusion_map[1:]:
        radian_distance = nextpoint[0] - point[0]
        if radian_distance > occlusion_resolution:
            new_map.append(point)
            point = tuple(nextpoint)
        elif radian_distance > -angle_acceptable_overlap:
            point = (point[0], point[1], nextpoint[2], point[3], nextpoint[4])
        else:
            raise Exception("overlap in occlusion map")
    new_map.append(point)
    occlusion_map = new_map
    
    # taking completely empty chunks and replacing with ground bounds
    # ground points may be absorbed, or ignored by preprocessing choice
    # previous codes include more absorption reasoning --- not using atm
    # aka some empty (high distance) regions may actual be absorbed detections
    if ground_present is not None:
        newmap = []
        prevpivot = occlusion_map[0]
        for seg_idx in range(1, len(occlusion_map)):
            nextpivot = occlusion_map[seg_idx]
            if prevpivot[2] != inf:
                newmap.append(prevpivot)
                prevpivot = nextpivot
                continue
            assert nextpivot[1]==inf
            first_seg = prevpivot[4]
            last_seg = nextpivot[3]
            assert first_seg < last_seg
            first_angle = prevpivot[0]
            last_angle = nextpivot[0]
            if first_angle-.05 < groundforlaser_angles[0]:
                newmap.append(prevpivot)
                prevpivot = nextpivot
                continue
            if last_angle+.049 > groundforlaser_angles[-1]:
                newmap.append(prevpivot)
                prevpivot = nextpivot
                continue
            groundfirstidx = np.searchsorted(groundforlaser_angles, first_angle-.05)
            groundlastidx = np.searchsorted(groundforlaser_angles, last_angle-.05)

            if ground_present[groundfirstidx] < inf:
                prevpivot = (prevpivot[:2] + (ground_present[groundfirstidx],)
                             + prevpivot[3:])
            for groundidx in range(groundfirstidx, groundlastidx):
                newpivot = (groundforlaser_angles[groundidx],
                            prevpivot[2], ground_present[groundidx+1],
                            first_seg+.5, first_seg+.5)
                if newpivot[2] < inf or newpivot[3] < inf:
                    newmap.append(prevpivot)
                    prevpivot = newpivot
                # otherwise, don't need to repeat infinite segments
            if (groundlastidx > len(ground_present) and
                                ground_present[groundlastidx] < inf):
                nextpivot = (nextpivot[:1] + (ground_present[groundlastidx],)
                             + nextpivot[2:])
            newmap.append(prevpivot)
            prevpivot = nextpivot
        newmap.append(prevpivot)
        occlusion_map = newmap
    
    return np.array(occlusion_map)



# line u,v vector with constant vx-uy=c
# ray goes from origin to x0,y0
def lineCrossesRay(u,v,c, x0,y0):
    cross = x0*v - y0*u
    # if cross ~= 0, line is parallel to ray
    if abs(cross) < 1e-5:
        return False, 0.
    posonray = c / cross
    # if c / cross < 0, line crosses behind origin wrt ray
    # if > 1, passes past x0,y0
    if posonray < 0 or posonray > 1:#.995:
        return False, 0.
    return True, (x0*u + y0*v)*posonray


""" how much can line extend in gap between lidar points?
for del_theta, u, v, extension del_v is
            (u^2 + v^2) sin(del_theta) / (ucos(dth)-vsin(dth))
"""
# multiply by two in case point outlier/failure
correction_angle_resolution = .0031 * 2
jump_at = series_distance_cutoff**.5
def _angleCorrection(u, v):
    u = abs(u)
    if u < 1e-3:
        if abs(v) < 1e-3:
            # this object is basically touching the origin... shouldn't happen
            return 0.
        return inf # very big
    else:
        return correction_angle_resolution*(u*u + v*v)/u

# 1/9/2019 same as above, but makes the angle a little more Xtreme
# 1/14/2019 accounts for segmentation strategy, high addition if jump > segment_cutoff
def angleCorrection(u, v):
    u = max(.001, abs(u)-.02)
    if u < 1e-2 and abs(v) < 1e-2:
        return 0. # object is basically touching the origin... shouldn't happen
    jump = correction_angle_resolution*(u*u+v*v)/u
    if jump > jump_at*2: # segmentation failure pretty much guaranteed
#        print("high jump")
        jump = inf
    elif jump > jump_at*1.5: # segmentation failure possible # EDIT 2/7/19 to 1.5
        jump += jump_at*2
    return jump

""" reverse of angleCorrection, assuming small angle again """
def angleGap(u, vlo, vhi):
    if vlo < 0: return 1. # high angle
    denom = u*u+vlo*vhi
    if denom <= 0: return 1.
    return (vhi-vlo)*u/denom    

visibility_cutoff = .003 # radians
def boundMeasurement(seg_idx, rec, occlusion_map):
    u, v, max_ulo, min_uhi, max_vlo, min_vhi = rec
    
    # if rec is too small, convert so it is straight
    # this way, behind-occlusion is still accounted for
    if min_uhi - max_ulo < .1 and min_vhi - max_vlo < .1:
        centeru = (max_ulo+min_uhi)/2
        centerv = (max_vlo+min_vhi)/2
        centerx = centeru*u+centerv*v
        centery = centeru*v-centerv*u
        uvlen = hypot(centerx, centery)
        rec = (-centery/uvlen, centerx/uvlen, -.05, .05, uvlen-.05, uvlen+.05)
        u, v, max_ulo, min_uhi, max_vlo, min_vhi = rec
    
    # full format = includes minimum and maximum bounds for each line
    min_ulo = -inf
    max_uhi = inf
    min_vlo = -inf
    max_vhi = inf
    
    # for occlusion purposes, find which sides are visible
    if max_ulo <= 0:
        uvisible = True
        vvisible = False
    else:
        angle_u = angleGap(max_vlo, max_ulo, min_uhi)
        angle_v = angleGap(max_ulo, max_vlo, min_vhi)
        if angle_u < angle_v and angle_u < visibility_cutoff:
            uvisible = False
            vvisible = True
        elif angle_v < visibility_cutoff:
            uvisible = True
            vvisible = False
        else:
            uvisible = True
            vvisible = True
    if uvisible:
        min_vlo = max_vlo
    if vvisible:
        min_ulo = max_ulo
    
    # convert occlusion map into form useful for bounding
    # the way we are currently bounding is:
    #   for each jump-segment (moving between occluding objects), see if this obj is visible
    #   only the further end of these jump segments is needed
    #   not using the actual object segments
    #   this will result in higher bounds, but never no bounds
    further_end = np.maximum(occlusion_map[:,1], occlusion_map[:,2])
    occlusion_map_points = np.array((np.cos(occlusion_map[:,0])*further_end,
                                     np.sin(occlusion_map[:,0])*further_end)).T
    
    # bound counter-clockwise first
    bounding_cw = True
    map_idx = np.searchsorted(occlusion_map[:,4], seg_idx)
    occ_x = np.cos(occlusion_map[map_idx,0])*occlusion_map[map_idx,1]
    occ_y = np.sin(occlusion_map[map_idx,0])*occlusion_map[map_idx,1]
    if vvisible:
        is_bound, bound = lineCrossesRay(v, -u, -max_ulo, occ_x, occ_y)
        if is_bound:
            max_vhi = min_vhi#bound
            bounding_cw = False
    else:
        is_bound, bound = lineCrossesRay(u, v, max_vlo, occ_x, occ_y)
        if is_bound:
            min_ulo = max_ulo#bound
            bounding_cw = False
#        if uvisible:
#            # 9/7/19 go ahead and do a simple bound on vhi using max_ulo (liberal)
#            # will get rid of extreme cases of lines seen as rectangles
#            is_bound, bound = lineCrossesRay(v, -u, -max_ulo, occ_x, occ_y)
#            if is_bound:
#                max_vhi = min_vhi
    # find this measurement's index in the occlusion map
    if bounding_cw and map_idx > 0:
        for occ_x, occ_y in occlusion_map_points[map_idx-1::-1]:
            # check occluded segment plane that extends in cw direction
            if vvisible:
                is_bound, bound = lineCrossesRay(v, -u, -max_ulo, occ_x, occ_y)
                if is_bound:
                    max_vhi = bound
                    # the first located boundary should be the closest
                    break
            else:
                is_bound, bound = lineCrossesRay(u, v, max_vlo, occ_x, occ_y)
                if is_bound:
                    min_ulo = bound
                    break
    # next bound clockwise
    bounding_cc = True
    map_idx = np.searchsorted(occlusion_map[:,3], seg_idx+.1)-1
    occ_x = np.cos(occlusion_map[map_idx,0])*occlusion_map[map_idx,2]
    occ_y = np.sin(occlusion_map[map_idx,0])*occlusion_map[map_idx,2]
    if uvisible:
        is_bound, bound = lineCrossesRay(u, v, max_vlo, occ_x, occ_y)
        if is_bound:
            max_uhi = min_uhi#bound
            bounding_cc = False
    else:
        is_bound, bound = lineCrossesRay(v, -u, -max_ulo, occ_x, occ_y)
        if is_bound:
            min_vlo = max_vlo#bound
            bounding_cc = False
#        if vvisible:
#            # 9/7/19 go ahead and do a simple bound on vhi using max_ulo (liberal)
#            # will get rid of extreme cases of lines seen as rectangles
#            is_bound, bound = lineCrossesRay(u, v, max_vlo, occ_x, occ_y)
#            if is_bound:
#                max_uhi = min_uhi
    if bounding_cc:
        for occ_x, occ_y in occlusion_map_points[map_idx+1:]:
            if uvisible:
                is_bound, bound = lineCrossesRay(u, v, max_vlo, occ_x, occ_y)
                if is_bound:
                    max_uhi = bound
                    break
            else:
                is_bound, bound = lineCrossesRay(v, -u, -max_ulo, occ_x, occ_y)
                if is_bound:
                    min_vlo = bound
                    break
        
    # if restraints in any direction are small, remove them
    if max_ulo - min_ulo < 0:
        min_ulo = max_ulo
    if max_uhi - min_uhi < 0:
        max_uhi = min_uhi
    if max_vlo - min_vlo < 0:
        min_vlo = max_vlo
    if max_vhi - min_vhi < 0:
        max_vhi = min_vhi
        
    # add corrections for finite lidar resolution
    if uvisible:
        max_uhi += angleCorrection(min_vlo, max_uhi)
    else:
        min_vlo -= angleCorrection(min_ulo, min_vlo)
    if vvisible:
        max_vhi += angleCorrection(min_ulo, max_vhi)
    else:
        min_ulo -= angleCorrection(min_vlo, min_ulo)
    
    # if restraints in any direction are small, remove them
    assert max_ulo >= min_ulo
    assert max_uhi >= min_uhi
    assert max_vlo >= min_vlo
    assert max_vhi >= min_vhi
        
    return (u,v, max_ulo, min_uhi, max_vlo, min_vhi,
            min_ulo, max_uhi, min_vlo, max_vhi)


"""
takes bounded rectangle, checks against car mold, and finds normal distribution
over xyalw parameters
"""
#variance_multiplier = 1.2
#msmt_noise = np.array((.2,.2,.25,.2,.2))
car_dims = ((2.95,4.9),(1.35,1.9),(1.25,2.))
car_dim_min_len, car_dim_max_len = car_dims[0][0]/2, car_dims[0][1]/2*1.2
car_dim_min_wid, car_dim_max_wid = car_dims[1][0]/2, car_dims[1][1]/2*1.2
def msmtBound2msmtNormal(rec):
    u,v,maxulo,minuhi,maxvlo,minvhi,minulo,maxuhi,minvlo,maxvhi = rec
    minulo = max(minulo, maxulo - 12) # otherwise, can have no idea where center is
    maxuhi = min(maxuhi, minuhi + 12)
    minvlo = max(minvlo, maxvlo - 12)
    maxvhi = min(maxvhi, minvhi + 12)
    uvT = np.array(((u,v),(v,-u)))
    uvworks = (maxuhi-minulo > car_dim_min_len*2 and
               minuhi-maxulo < car_dim_max_len*2 and
               maxvhi-minvlo > car_dim_min_wid*2 and 
               minvhi-maxvlo < car_dim_max_wid*2)
    if uvworks:
        umean, ucov = uvBound2xyNormal(minulo,maxulo,minuhi,maxuhi,
                                       car_dim_min_len, car_dim_max_len)
        vmean, vcov = uvBound2xyNormal(minvlo,maxvlo,minvhi,maxvhi,
                                       car_dim_min_wid, car_dim_max_wid)
        xymean = np.array((umean[0]*u+vmean[0]*v, umean[0]*v-vmean[0]*u,
                           np.arctan2(v,u), umean[1], vmean[1]))
        xycov = np.zeros((5,5))
        xycov[[0,0,3,3],[0,3,0,3]] = ucov.reshape((-1,))
        xycov[[1,1,4,4],[1,4,1,4]] = vcov.reshape((-1,))
        xycov[:2,:] = uvT.dot(xycov[:2,:])
        xycov[:,:2] = xycov[:,:2].dot(uvT.T)
        uvout = (xymean, xycov)
    else:
        uvout = None
    vuworks = (maxvhi-minvlo > car_dim_min_len*2 and
               minvhi-maxvlo < car_dim_max_len*2 and
               maxuhi-minulo > car_dim_min_wid*2 and 
               minuhi-maxulo < car_dim_max_wid*2)
    if vuworks:
        vmean, vcov = uvBound2xyNormal(minvlo,maxvlo,minvhi,maxvhi,
                                       car_dim_min_len, car_dim_max_len)
        umean, ucov = uvBound2xyNormal(minulo,maxulo,minuhi,maxuhi,
                                       car_dim_min_wid, car_dim_max_wid)
        xymean = np.array((umean[0]*u+vmean[0]*v, umean[0]*v-vmean[0]*u,
                           np.arctan2(-u,v), vmean[1], umean[1]))
        xycov = np.zeros((5,5))
        xycov[[0,0,4,4],[0,4,0,4]] = ucov.reshape((-1,))
        xycov[[1,1,3,3],[1,3,1,3]] = vcov.reshape((-1,))
        xycov[:2,:] = uvT.dot(xycov[:2,:])
        xycov[:,:2] = xycov[:,:2].dot(uvT.T)
        vuout = (xymean, xycov)
    else:
        vuout = None
    return uvout, vuout


"""
look up point height in ground grid
if point z not given, return elevation (height from cam pov at ground)
else return height above ground
"""
def getGrndHeight(ground, x, y, z=None):
    tilex = min(max(int(x/grndstep[0])-grndstart[0], 0), grndlen[0]-1)
    tiley = min(max(int(y/grndstep[1])-grndstart[1], 0), grndlen[1]-1)
    tile = ground[tilex,tiley]
    if z is None:
        return tile[3] - tile[0]*x - tile[1]*y
    else:
        return tile[0]*x+tile[1]*y+tile[2]*z-tile[3]
    

" lidar properties "
laser_angles = np.arange(0., 64)
laser_angles[:32] = -.33333*laser_angles[:32] + 3.
laser_angles[32:] = -.5*laser_angles[32:] + laser_angles[31] + 31*.5
laser_angles = np.tan(laser_angles * np.pi/180.)
#laser_angles = laser_angles[lasers]
laser_angles += -.02 # correction for observed angles
laser_intercepts = np.zeros(64)
laser_intercepts[:32] = .209 - .00036*np.arange(32,dtype=float)
laser_intercepts[32:] = .126 - .00032*np.arange(32,dtype=float)
laser_intercepts += 1.65
# laser angle space * current laser gap * some multiplier for missing lasers
height_angle_slack = .01*4*2
lasers = list(range(55))
laser_angles = laser_angles[lasers]
laser_intercepts = laser_intercepts[lasers]

## quanergy
#lasers2use = [0,6,15,25,33,39,45,50]
# more practical for camera fusion case, doesn't cover nearby objects that are easy
#lasers2use = [1,7,13,19,25]#,31,37]
lasers2use = []#[1,6,11,16,21,26]


"""
make sets of normally-distributed measurements from lidar
data = nX3 array as from kitti dataset
calib = 4x4 lidar-to-camera transformation matrix
ground = grid (size specified in config) of planes
view = tangent (lat/lon) of image view, or desired lidar view
"""
mindetectedlength4box = 1.
def processLidar(data, calib, ground, view, laser2use):
    
#    # get angles of all ground points
#    # this is cheating and using all 64 lasers atm
#    # but making it use only available lasers requires two separate loops...
#    # also consider not using pt-based check for ground in occlusion, just pass []
#    heights = data.dot(calib[2,:3]) + calib[2,3]
#    groundpts = (heights < .1) & (data[:,0] > .1)
#    groundptangles = np.arctan2(data[groundpts,1], data[groundpts,0])
#    groundptangles.sort()

    # separate points by laser
    # this will be repeated for every laser, and is pretty slow...
    # easy target for speedup if necessary
    starts = np.where(np.diff(np.sign(data[:,1])) > 0)[0]
    starts = np.concatenate(([0], starts+1, [len(data)]))
    true_starts = np.append(np.diff(starts) > 2, [True])
    starts = starts[true_starts]
    assert starts.shape[0] > 55
    
    # subselect useful data from laser
    pts = data[starts[laser2use]:starts[laser2use+1]]
    include = pts[:,0] > 0
    include &= abs(pts[:,1]) < pts[:,0]*view + 2.
    include &= pts.dot(calib[2,:3]) + calib[2,3] > -.3
    pts = pts[include]
    # ensure sweep is contiguous
    swap_idx = np.where(np.diff(np.arctan2(pts[:,1],pts[:,0]))<-.05)[0]
    assert len(swap_idx) <= 1
    if len(swap_idx) == 1:
        swap_idx = swap_idx[0] + 1
        pts = np.append(pts[swap_idx:], pts[:swap_idx], axis=0)

    segs = segmentPoints(pts)
    # classify segments by ground, worth using, etc
    msmts = []
    segsareground = []
    segsinclude = []
    for segidx, seg in enumerate(segs):
        segmiddle = np.mean(seg,axis=0)
        segmiddle = calib[:3,:3].dot(segmiddle) + calib[:3,3]
        seggroundelev = getGrndHeight(ground, segmiddle[0], segmiddle[1])
        heights = seg.dot(calib[2,:3])+calib[2,3]
        segisground = max(heights)-seggroundelev < .3
        seginclude = (not segisground) and segmiddle[2]-seggroundelev < 2.
        ## 9/17/19 LAST DITCH
        #seginclude = (not segisground) and segmiddle[2]-seggroundelev < 1.5
        segsareground.append(segisground)
        segsinclude.append(seginclude)
        msmts.append(makeMeasurement(seg))
    # calculate/approximate occlusion along this laser sweep
    ground_present = getGroundForLaser(ground, laser_angles[laser2use])
    starting_angle = min(-view, np.arctan2(pts[0,1],pts[0,0])) # was view-.1
    ending_angle = max(view, np.arctan2(pts[-1,1],pts[-1,0])) # put outside call
    occlusion_map = makeOcclusionMap(segs, segsareground, msmts,
                        starting_angle, ending_angle, True, ground_present)
    
    boxmsmts = []
    fragmsmts = []
    for seg_idx in range(len(segs)):
        if not segsinclude[seg_idx]:
            continue
        msmt = boundMeasurement(seg_idx, msmts[seg_idx], occlusion_map)
        if max(msmt[5]-msmt[4],msmt[3]-msmt[2]) < mindetectedlength4box:
            msmt1,msmt2 = None,None
        else:
            msmt1, msmt2 = msmtBound2msmtNormal(msmt)
        if msmt1 is not None:
            mean, cov = msmt1
            mean[:2] = calib[:2,:2].dot(mean[:2]) + calib[:2,2]
            mean[2] += np.arctan2(calib[1,0],calib[0,0])
            cov[:2,:] = calib[:2,:2].dot(cov[:2,:])
            cov[:,:2] = cov[:,:2].dot(calib[:2,:2].T)
            boxmsmts.append((mean, cov))
        if msmt2 is not None:
            mean, cov = msmt2
            mean[:2] = calib[:2,:2].dot(mean[:2]) + calib[:2,2]
            mean[2] += np.arctan2(calib[1,0],calib[0,0])
            cov[:2,:] = calib[:2,:2].dot(cov[:2,:])
            cov[:,:2] = cov[:,:2].dot(calib[:2,:2].T)
            boxmsmts.append((mean, cov))
        if msmt1 is None and msmt2 is None:
            # make a misc msmt
            # which is only position
            # 9/3/19 only include too-small fragments, not too-large ones
            nottoobig = ((msmt[3]-msmt[2] < car_dim_max_len*2) and
                         (msmt[5]-msmt[4] < car_dim_max_wid*2)) or (
                         (msmt[3]-msmt[2] < car_dim_max_wid*2) and
                         (msmt[5]-msmt[4] < car_dim_max_len*2))
            # 9/15/19 don't include things below a certain size
            nottoosmall = max(msmt[5]-msmt[4],msmt[3]-msmt[2]) > .4
            if nottoobig and nottoosmall:
                mean = calib[:2,:2].dot(uv2xy(msmt[:6])[:2])+calib[:2,2]
                fragmsmts.append(mean)
            
    # find minimum distances at which cars might be hit
    # at closer distances, lidar is too high
    detectzone = getGroundForLaser(ground, laser_angles[laser2use],
                                   laser_height=1.65-1.2)
    return (detectzone, occlusion_map[:,:3], boxmsmts, fragmsmts)