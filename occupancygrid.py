#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 7/1/19
using opencv's warpAffine instead of handmade grid conversions
"""
import numpy as np
from cv2 import warpAffine, BORDER_TRANSPARENT, BORDER_CONSTANT
from cv2 import INTER_LINEAR, INTER_CUBIC, WARP_INVERSE_MAP


tile_size = .5
tile_start = (-10., -50.)
occupancygrid_shape = (140, 200)

ntx, nty = occupancygrid_shape
grid = np.mgrid[:ntx, :nty]
grid = grid.transpose((1,2,0)).astype(float)
#grid += .5
grid *= tile_size
grid += tile_start


def updateGridwGrid(priorgrid, msmtgrid, viewedgrid, msmt_confusion_matrix):
    """
    given an occupancy prior and measured occupancy, update posterior of occupancy
    prior and posterior are float matrices
    msmtgrid and viewedgrid are boolean matrices
    if there is a msmt in the grid, it is a positive measurement
    if there is no msmt and the tile was viewed, it is a negative measurement
    if the tile is not viewed it is not updated
    P(x=1|z=1) = P(x=1)P(z=1|x=1)/(P(x=1)P(z=1|x=1) + (1-P(x=1))P(z=1|x=0))
    """
    tnp,fpp = msmt_confusion_matrix[0]
    fnp,tpp = msmt_confusion_matrix[1]
    posterior_seen = priorgrid*tpp/(fpp + priorgrid*(tpp-fpp))
    posterior_notseen = priorgrid*fnp/(tnp + priorgrid*(fnp-tnp))
    posterior = priorgrid.copy()
    posterior[msmtgrid] = posterior_seen[msmtgrid]
    notseen = (msmtgrid==False) & viewedgrid
    posterior[notseen] = posterior_notseen[notseen]
    return posterior
    
#
#def reOrientGridOlder(priorgrid, transform, initial_val, gridstep, gridstart, gridlen):
#    """
#    transform = [[cos,-sin,tx],[sin,cos,ty],[0,0,1]]
#    shift an occupancy grid
#    Obviously, there is error due to imperfect matching of old and new tiles.
#    This function does an approximation by finding the old tiles corresponding to
#    the bottom left and top right corners of the new tile. The returned occupancy
#    is a weighted sum of the two. Tiles that were previously out of frame are
#    set to the initial value.
#    """
#    grid = np.mgrid[gridstart[0]:gridstart[0]+gridlen[0],
#                    gridstart[1]:gridstart[1]+gridlen[1]]
#    grid = grid.transpose((1,2,0)) * gridstep
#    tile_distance_limit = gridstep[0]*2**.5 + .01
#    newgrid_pos = (grid - transform[:2,2]).dot(transform[:2,:2])
#    newgrid_idxs = np.floor(newgrid_pos/gridstep).astype(int) - gridstart
#    newgrid_bl_xidx = newgrid_idxs[:,:,0]
#    newgrid_bl_yidx = newgrid_idxs[:,:,1]
#    newgrid_outofzone  = newgrid_bl_xidx >= gridlen[0]
#    newgrid_outofzone |= newgrid_bl_yidx >= gridlen[1]
#    newgrid_outofzone |= newgrid_bl_xidx < 0
#    newgrid_outofzone |= newgrid_bl_yidx < 0
#    newgrid_idxs[newgrid_outofzone] = 0
#    newgrid_diff = newgrid_pos - grid[newgrid_bl_xidx,newgrid_bl_yidx]
#    newgrid_bl_score = tile_distance_limit - np.hypot(newgrid_diff[:,:,0],
#                                                      newgrid_diff[:,:,1])
#    newgrid_bl_score[newgrid_outofzone] = 0
#    assert np.all(newgrid_bl_score >= 0)
#    
#    newgrid_pos = (grid + gridstep - transform[:2,2]).dot(transform[:2,:2])
#    newgrid_idxs = np.floor(newgrid_pos/gridstep).astype(int) - gridstart
#    newgrid_tr_xidx = newgrid_idxs[:,:,0]
#    newgrid_tr_yidx = newgrid_idxs[:,:,1]
#    newgrid_outofzone  = newgrid_tr_xidx >= gridlen[0]
#    newgrid_outofzone |= newgrid_tr_yidx >= gridlen[1]
#    newgrid_outofzone |= newgrid_tr_xidx < 0
#    newgrid_outofzone |= newgrid_tr_yidx < 0
#    newgrid_idxs[newgrid_outofzone] = 0
#    newgrid_diff = grid[newgrid_tr_xidx,newgrid_tr_yidx]+gridstep - newgrid_pos
#    newgrid_tr_score = tile_distance_limit - np.hypot(newgrid_diff[:,:,0],
#                                                      newgrid_diff[:,:,1])
#    newgrid_tr_score[newgrid_outofzone] = 0
#    assert np.all(newgrid_tr_score >= 0)
#    
#    newgrid = newgrid_bl_score * priorgrid[newgrid_bl_xidx, newgrid_bl_yidx]
#    newgrid += newgrid_tr_score * priorgrid[newgrid_tr_xidx, newgrid_tr_yidx]
#    newgrid += initial_val * .001
#    newgrid_count = newgrid_bl_score + newgrid_tr_score + .001
#    newgrid /= newgrid_count
#    return newgrid


def reOrientGridOld(priorgrid, transform, initial_val, gridstep, gridstart, gridlen):
    """
    transform = [[cos,-sin,tx],[sin,cos,ty],[0,0,1]]
    shift an occupancy grid
    Obviously, there is error due to imperfect matching of old and new tiles.
    This function does an approximation by finding the old tiles corresponding to
    points evenly spaced within the tile
    """
    r = 4
    grid = np.mgrid[gridstart[0]:gridstart[0]+gridlen[0],
                    gridstart[1]:gridstart[1]+gridlen[1]]
    grid = grid.transpose((1,2,0)) * gridstep
    jumps = np.mgrid[:r,:r].transpose((1,2,0)).reshape((r**2,2))
    jumps = (jumps + .5)*(gridstep/r)
    newgrid = np.zeros(gridlen)
    for jump in jumps:
        newgrid_pos = (grid + jump - transform[:2,2]).dot(transform[:2,:2])
        newgrid_idxs = np.floor(newgrid_pos/gridstep).astype(int) - gridstart
        newgrid_xidx = newgrid_idxs[:,:,0]
        newgrid_yidx = newgrid_idxs[:,:,1]
        newgrid_inzone  = ((newgrid_xidx < gridlen[0]) &
                           (newgrid_yidx < gridlen[1]) &
                           (newgrid_xidx >= 0) &
                           (newgrid_yidx >= 0))
        newgrid_idxs[newgrid_inzone==False] = 0 # doesn't matter, just avoids IndexError
        newgrid += np.where(newgrid_inzone, priorgrid[newgrid_xidx, newgrid_yidx],
                            initial_val)
    newgrid /= r**2
    return newgrid

def reOrientGrid(priorgrid, transform, initial_val, gridstep, gridstart, gridlen):
    """
    transform = [[cos,-sin,tx],[sin,cos,ty],[0,0,1]]
    shift an occupancy grid
    Obviously, there is error due to imperfect matching of old and new tiles.
    This function does an approximation by finding the old tiles corresponding to
    points evenly spaced within the tile
    """
    movex = -gridstart[0]+transform[0,2]/gridstep[0]
    movex += transform[0,1]*gridstart[1] + transform[0,0]*gridstart[0]
    movey = -gridstart[1]+transform[1,2]/gridstep[1]
    movey += transform[1,1]*gridstart[1] + transform[1,0]*gridstart[0]
    T = np.array([[transform[1,1],transform[1,0],movey],
                  [transform[0,1],transform[0,0],movex]])
    return warpAffine(priorgrid, T, (gridlen[1],gridlen[0]),
                      flags=INTER_LINEAR,#+WARP_INVERSE_MAP,
                      borderMode=BORDER_CONSTANT, borderValue=initial_val)

    
    
    
#from groundRat import getElevation

import numba as nb
@nb.jit(nb.void(nb.b1[:,:], nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8))
def fillTriangle(canvas, ax, ay, bx, by, cx, cy):
    """
    helper for gridViewable
    based on 
    http://www-users.mat.uni.torun.pl/~wrona/3d_tutor/tri_fillers.html
    assumes a->b->c is ccw, ax<bx, ax<cx
    """
    dxb = (by-ay)/(bx-ax)
    dxc = (cy-ay)/(cx-ax)
    if cx > bx + 1:
        secondx = int(bx)
        thirdx = int(cx)
        dxb2 = (cy-by)/(cx-bx)
        dxc2 = dxc
    elif bx > cx + 1:
        secondx = int(cx)
        thirdx = int(bx)
        dxb2 = dxb
        dxc2 = (by-cy)/(bx-cx)
    else:
        secondx = int(bx)
        thirdx = secondx
        dxb2 = 0.
        dxc2 = 0.
    syb = ay
    syc = ay
    for sx in range(int(ax), secondx):
        syb += dxb
        syc += dxc
        canvas[sx, int(syb):int(syc)] = True
    for sx in range(secondx, thirdx):
        syb += dxb2
        syc += dxc2
        canvas[sx, int(syb):int(syc)] = True

extradist = 1.
def gridViewable(occlusion_maps, occlusion_info, ground):
    """
    go faster -> triangle filling
    """
    gridcenters = grid + tile_size/2.
    dists = np.hypot(gridcenters[:,:,1], gridcenters[:,:,0])
    heights = ground[:,:,3]-ground[:,:,0]*gridcenters[:,:,0]-ground[:,:,1]*gridcenters[:,:,1]
    #heights = getElevation(gridcenters, ground)
    effective_min_angle = (0-1.65+heights)/dists
    effective_max_angle = (2-1.65+heights)/dists
    origin_x = np.searchsorted(gridcenters[:,0,0], 0)
    origin_y = np.searchsorted(gridcenters[0,:,1], 0)
    max_x = tile_start[0]+tile_size*ntx
    max_y = tile_start[1]+tile_size*nty
    
    viewed = np.zeros(occupancygrid_shape, dtype=bool)
    occlusion_startidx = 0
    for occlusion_endidx, laser_angle in occlusion_info:
        occlusion_endidx = int(occlusion_endidx)
        if occlusion_startidx == occlusion_endidx:
            continue
        included = effective_min_angle < laser_angle
        included &= effective_max_angle > laser_angle
        inzone = np.zeros(occupancygrid_shape, dtype=bool)
        starting_angle = occlusion_maps[occlusion_startidx,0]
        starting_dist = occlusion_maps[occlusion_startidx,2]
        for ending_angle,ending_dist,nextdist in\
                    occlusion_maps[occlusion_startidx+1:occlusion_endidx]:
            if starting_dist > 2:
                c = np.cos(starting_angle)
                s = np.sin(starting_angle)
                dist = starting_dist + extradist
                if dist*c > max_x:
                    dist = max_x/c
                if dist*s > max_y:
                    dist = max_y/s
                if dist*s < tile_start[1]:
                    dist = tile_start[1]/s
                starting_x = (c*dist - tile_start[0]) / tile_size
                starting_y = (s*dist - tile_start[1]) / tile_size
                c = np.cos(ending_angle)
                s = np.sin(ending_angle)
                dist = ending_dist + extradist
                if dist*c > max_x:
                    dist = max_x/c
                if dist*s > max_y:
                    dist = max_y/s
                if dist*s < tile_start[1]:
                    dist = tile_start[1]/s
                ending_x = (c*dist - tile_start[0]) / tile_size
                ending_y = (s*dist - tile_start[1]) / tile_size
                fillTriangle(inzone, origin_x, origin_y,
                             starting_x, starting_y, ending_x, ending_y)
            starting_dist = nextdist
            starting_angle = ending_angle
        viewed |= included & inzone
        occlusion_startidx = occlusion_endidx
    return viewed


def mixGrid(grid, mixer, outervalue, tempmat=None):
    """
    perform 2d convolution on a grid to emulate propagation between adjacent tiles
    tiles outside the limit of the grid are set to outervalue
    """
    assert mixer.shape[0]%2 and mixer.shape[1]%2
    pad = np.array(mixer.shape)//2
    if tempmat is None:
        tempmat = np.zeros(grid.shape+pad*2, dtype=grid.dtype)
    else:
        assert all(pad*2+grid.shape == tempmat.shape)
    tempmat[pad[0]:-pad[0], pad[1]:-pad[1]] = grid
    tempmat[:pad[0],:] = outervalue
    tempmat[-pad[0]:,:] = outervalue
    tempmat[:,:pad[1]] = outervalue
    tempmat[:,-pad[1]:] = outervalue
    
    viewshape = (grid.shape[0], grid.shape[1], mixer.shape[0], mixer.shape[1])
    view4conv = np.lib.stride_tricks.as_strided(tempmat, viewshape,
                                                tempmat.strides*2, writeable=False)
    grid[:] = np.einsum(view4conv, [0,1,2,3], mixer, [2,3], [0,1])
    
    
    
""" useful polynomial approximation of normal cdf
    source = John D Cooke blog """
def approxNormalCdf(dev): return 1./(1 + np.exp(-.07056 * dev**3 - 1.5976 * dev))
    
def eigTwoxTwo(varx, vary, covxy):
    # from math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
    T = (varx+vary)*.5
    D = varx*vary-covxy*covxy
    eigval1 = T + np.sqrt(T*T-D)
    eigval2 = 2*T - eigval1
    eigvecnorm = np.hypot(eigval1-vary, covxy)
    return eigval1, eigval2, (eigval1-vary)/eigvecnorm, covxy/eigvecnorm

""" this is an approximate cdf, assuming independent prob in x and y directions
"""
def mapNormal2Grid(meanx, meany, varx, vary, covxy,
                    gridstart, gridstep, gridlen):
    xposs = np.arange(gridstart[0], gridstart[0]+gridlen[0]+1) * gridstep[0]
    cdf = approxNormalCdf((xposs-meanx) / varx**.5)
    totalprobinside = cdf[-1] - cdf[0]
    if totalprobinside < 1e-10:
        # very low likelihood of appearance, just set to uniform
        return np.zeros(gridlen) + 1./gridlen[0]/gridlen[1]
    llx = np.diff(cdf) / totalprobinside
    yposs = np.arange(gridstart[1], gridstart[1]+gridlen[1]+1) * gridstep[1]
    cdf = approxNormalCdf((yposs-meany) / vary**.5)
    totalprobinside = cdf[-1] - cdf[0]
    if totalprobinside < 1e-10:
        return np.zeros(gridlen) + 1./gridlen[0]/gridlen[1]
    lly = np.diff(cdf) / totalprobinside
    return np.outer(llx, lly)

""" approximate cdf, accounting for xy correlation
make normal cdf grid, do rotation transform to put on rectified grid
don't do scale transform, b.c. you would need to sum probs for increased scale
note: cv2 transformation matrices have y-axis first
"""
def mapNormal2GridRot(meanx, meany, varx, vary, covxy,
                    gridstart, gridstep, gridlen):
    rectvx, rectvy, rectc, rects = eigTwoxTwo(varx, vary, covxy)
    gridcenter = (gridstart + gridlen*.5)*gridstep
    rotmeanx = rectc*(meanx-gridcenter[0]) + rects*(meany-gridcenter[1]) + gridcenter[0]
    rotmeany = rectc*(meany-gridcenter[1]) - rects*(meanx-gridcenter[0]) + gridcenter[1]
    ingrid = mapNormal2Grid(rotmeanx, rotmeany, rectvx, rectvy,
                            0, gridstart, gridstep, gridlen)
    midx, midy = gridlen*.5 - .5
    T = np.array(((rectc,  rects, midy-rectc*midy-rects*midx),
                  (-rects, rectc, midx-rectc*midx+rects*midy)))
    outgrid = warpAffine(ingrid, T, (gridlen[1], gridlen[0]),
                         flags=INTER_LINEAR,
                         borderMode=BORDER_CONSTANT, borderValue=0.)
    # bilinear interpolation may alter the sum of values
    # problem for probability distributions
    if np.sum(outgrid) > 1: outgrid /= np.sum(outgrid)
    return outgrid
    
"""
return subgrid with probability of occupancy
and subgrid location
default subsize 0 -- just pick the center cell and return this
if cell outside of grid, returns size-0 subgrid
"""
def mapNormal2Subgrid(normalparams, gridstart, gridstep, gridlen, subsize = 0):
    meanx, meany, varx, vary, covxy = normalparams
    tilex = int(np.floor(meanx/gridstep[0]))-gridstart[0]
    tiley = int(np.floor(meany/gridstep[1]))-gridstart[1]
    tilexmin = max(tilex-subsize, 0)
    tilexmax = min(tilex+subsize+1,gridlen[0])
    tileymin = max(tiley-subsize, 0)
    tileymax = min(tiley+subsize+1,gridlen[1])
    subgridstart = np.array((tilexmin, tileymin))
    subgridlen = np.array((tilexmax-tilexmin, tileymax-tileymin))
    if any(subgridlen <= 0): # size-0 subgrid
        return np.array((0,0)), np.zeros((0,0))
    subgrid = mapNormal2GridRot(meanx, meany, varx, vary, covxy,
                                subgridstart + gridstart, gridstep, subgridlen)
    return subgridstart, subgrid
    
    
    
"""
test reorientation, normal mapping, and mixing
"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ioff()
    
    gridstart = np.array((-4,-2))
    gridlen = np.array((8,8))
    gridstep = np.array((3.,3.))
    
    
    meanx = -3.
    meany = 8.
    varx = 4.**2
    vary = 3.**2
    covxy = .2*4*3
    normalmean = np.array((meanx, meany))
    normalvar = np.array(((varx, covxy), (covxy, vary)))
    
    ## make a high-res mesh of the normal distribution
    hresgridx, hresgridy = np.meshgrid(np.linspace(-12., 12, 100),
                                       np.linspace(-6., 18, 100))
    precvals, precvec = np.linalg.eigh(normalvar)
#    rectvarx, rectvary, precvecx, precvecy = eigTwoxTwo(4., 4., -1.)
#    precvals = np.array((rectvarx, rectvary))
#    precvec = np.array(((precvecx, -precvecy),(precvecy, precvecx)))
    precvals = 1/precvals
    ll1 = precvals[0]*(precvec[0,0]*(hresgridx-normalmean[0]) +
                       precvec[1,0]*(hresgridy-normalmean[1]))
    ll2 = precvals[1]*(precvec[0,1]*(hresgridx-normalmean[0]) +
                       precvec[1,1]*(hresgridy-normalmean[1]))
    ll = np.exp(-.5*(ll1*ll1 + ll2*ll2))
    
    ## map the normal distribution to the grid
    outgrid = mapNormal2GridRot(meanx, meany, varx, vary, covxy,
                                gridstart, gridstep, gridlen)
    
    ## compare mapped distribution to correct version
    plt.subplot(121).contour(hresgridx, hresgridy, ll)
    outgridForShow = outgrid.T[::-1]
    plt.subplot(122).imshow(outgridForShow)
    plt.show()
    
#    ## re-orient distribution
    #transform = np.array(((1.,0,-4),(0,1,0),(0,0,1)))
    transform = np.array(((.9798, -.2, -2.), (.2, .9798, 0), (0,0,1)))
    initial_val = .1
    reoriented1 = reOrientGridOld(outgrid, transform, initial_val,
                               gridstep, gridstart, gridlen)
    reoriented2 = reOrientGrid(outgrid, transform, initial_val,
                                gridstep, gridstart, gridlen)
    plt.figure(figsize=(10.,8.))
    plt.subplot(221).imshow(outgrid.T[::-1])
    plt.subplot(223).imshow(reoriented1.T[::-1])
    plt.subplot(224).imshow(reoriented2.T[::-1])
    plt.show()