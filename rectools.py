#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last mod 8/21/19
useful functions involving rectangles
adapted from previous rectangle.py, molds.py, and segment.py

xyalw rectangle (or xy for short)
uv rectangle
bounded uv rectangle
"""

from math import atan2, hypot, cos, sin
import numpy as np


" convert uv format to xyalw format"
def uv2xy(rec):
    u,v,ulo,uhi,vlo,vhi = rec[:6]
    uc = (ulo+uhi)/2.
    vc = (vlo+vhi)/2.
    return (u*uc+v*vc, v*uc-u*vc, atan2(v,u), (uhi-ulo)/2., (vhi-vlo)/2.)

"""
set up rectangle s.t. vlo > 0 (and if possible ulo > 0)
means that ulo,vlo is the visible corner, and 3 corners are visible if ulo>0
"""
def standardize(rec):
    u,v,ulo,uhi,vlo,vhi = rec
    assert uhi > ulo and vhi > vlo
    if vlo > 0:
        return rec if uhi > 0 else (v,-u,vlo,vhi,-uhi,-ulo)
    elif vhi < 0:
        return (-u,-v,-uhi,-ulo,-vhi,-vlo) if ulo<0 else (-v,u,-vhi,-vlo,ulo,uhi)
    elif ulo > 0:
        return (-v,u,-vhi,-vlo,ulo,uhi)
    elif uhi < 0:
        return (v,-u,vlo,vhi,-uhi,-ulo)
    else:
        raise ValueError("rectangle covers origin")

" convert xyalw format to uv format "
def xy2uv(rec):
    x,y,a,l,w = rec
    u,v = cos(a), sin(a)
    centeru = x*u + y*v
    centerv = x*v - y*u
    return standardize((u,v,centeru-l,centeru+l,centerv-w,centerv+w))
    
    
"""
transform uniform distribution over rectangle bounds to normal distribution
on xyalw rectangle format
angle's variance is set to 0
TODO check that you are dividing by 2 correctly
"""
def uvUniform2xyNormal(msmt):
    u,v,maxulo,minuhi,maxvlo,minvhi,minulo,maxuhi,minvlo,maxvhi = msmt
    meanuv = np.array((minulo+maxulo,minuhi+maxuhi,minvlo+maxvlo,minvhi+maxvhi))/2
    varuv = np.array((maxulo-minulo,maxuhi-minuhi,maxvlo-minvlo,maxvhi-minvhi))
    varuv = varuv**2 / 12.
    uv2xyTM = np.array(((u,u,v,v),(v,v,-u,-u),(0,0,0,0),(-1,1,0,0),(0,0,-1,1)))/2
    xy_mean = uv2xyTM.dot(meanuv)
    xy_mean[2] = atan2(v,u)
    xy_cov = (uv2xyTM * varuv).dot(uv2xyTM.T)
    return xy_mean, xy_cov
    

"""
determines whether bounded uv rectangle could be a car based on size
if possible, returns tighter bounds fitting car shape
if not returns None
"""
car_dims = ((2.95,4.9),(1.35,1.9),(1.25,2.)) # 99th percentile for kitti scored cars
car_dim_minlen = car_dims[0][0]
car_dim_maxlen = car_dims[0][1]*1.2
car_dim_minwid = car_dims[1][0]
car_dim_maxwid = car_dims[1][1]*1.2
def fitRecToMold(rec):
    u,v,maxulo,minuhi,maxvlo,minvhi,minulo,maxuhi,minvlo,maxvhi = rec
    # car where u,v is heading
    erru, ulo, uhi = _fitInterval(minulo, maxulo, minuhi, maxuhi,
                                  car_dim_minlen, car_dim_maxlen)
    errv, vlo, vhi = _fitInterval(minvlo, maxvlo, minvhi, maxvhi,
                                  car_dim_minwid, car_dim_maxwid)
    if erru < 0 and errv < 0:
        standardcar = (u,v,ulo,uhi,vlo,vhi)
    else:
        standardcar = None
    # car where v,-u is heading
    erru, ulo, uhi = _fitInterval(minulo, maxulo, minuhi, maxuhi,
                                  car_dim_minwid, car_dim_maxwid)
    errv, vlo, vhi = _fitInterval(minvlo, maxvlo, minvhi, maxvhi,
                                  car_dim_minlen, car_dim_maxlen)
    if erru < 0 and errv < 0:
        tcar = (u,v,ulo,uhi,vlo,vhi)
    else:
        tcar = None
    return standardcar, tcar

"""
a crazy function that takes rectangle bounds on lo and hi parameters,
and length bounds (hi-lo), and finds the polygon of possible lo-hi values
return polygon (lo,hi) corners in cc order
ultimately abandoned -- getting covariance of poygon would be a huge pain
"""
def _fitInterval(minlo,maxlo,minhi,maxhi, minlen,maxlen):
    # helper function bounds rectangle in one direction (interval)
    assert maxhi-minlo > minhi-maxlo and maxlen > minlen
    # corners sorted by length
    if minhi-minlo > maxhi-maxlo:
        currentcorners = [(minhi,maxlo),(maxhi,maxlo),(minhi,minlo),(maxhi,minlo)]
    else:
        currentcorners = [(minhi,maxlo),(minhi,minlo),(maxhi,maxlo),(maxhi,minlo)]
    if minlen > maxhi-minlo:
        crop = minlen - (maxhi-minlo)
        return crop, (minlo-crop/2,maxhi+crop/2)
    elif minlen > maxhi-maxlo and minlen > minhi-minlo:
        locrop = 3
        lowercorners = ((minlo+minlen,minlo), (maxhi,maxhi-minlen))
    elif minlen > maxhi-maxlo:
        locrop = 2
        lowercorners = ((minlo+minlen,minlo), (maxlo+minlen,maxlo))
    elif minlen > minhi-minlo:
        locrop = 2
        lowercorners = ((minhi,minhi-minlen), (maxhi,maxhi-minlen))
    elif minlen > minhi-maxlo:
        locrop = 1
        lowercorners = ((minhi,minhi-minlen), (maxlo+minlen,maxlo))
    else:
        locrop = 0
        lowercorners = []
    if maxlen < minhi-maxlo:
        crop = minhi-maxlo - maxlen
        return crop, (maxlo+crop/2,minhi-crop/2)
    elif maxlen < maxhi-maxlo and maxlen < minhi-minlo:
        hicrop = 1
        uppercorners = ((maxlo+maxlen,maxlo), (minhi,minhi-maxlen))
    elif maxlen < maxhi-maxlo:
        hicrop = 2
        uppercorners = ((maxhi,maxhi-maxlen), (minhi,minhi-maxlen))
    elif maxlen < minhi-minlo:
        hicrop = 2
        uppercorners = ((maxlo+maxlen,maxlo), (minlo+maxlen,minlo))
    elif maxlen < minhi-maxlo:
        hicrop = 3
        uppercorners = ((maxhi,maxhi-maxlen), (minlo+maxlen,minlo))
    else:
        hicrop = 4
        uppercorners = []
    corners = lowercorners + currentcorners[locrop:hicrop] + uppercorners
    corners = np.array([(lo,hi) for hi,lo in corners])
    # sort again
    return 0., corners

"""
converting lo-hi bounds to normal dist first, then soft update with size bounds
"""
def uvBound2xyNormal(minlo,maxlo,minhi,maxhi, minlen,maxlen):
    varlo = (maxlo-minlo)**2 / 12
    varhi = (maxhi-minhi)**2 / 12
    meanmid = (minlo+maxlo+minhi+maxhi)/4
    varmid = varlo/4+varhi/4
    meanlen = (minhi+maxhi-minlo-maxlo)/4
    varlen = varhi/4+varlo/4
    cov = varhi/4-varlo/4
    meanmsmt = (minlen+maxlen)/2
    varmsmt = (maxlen-minlen)**2 / 12
    prec = 1./(varlen+varmsmt)
    postmeanmid = meanmid + cov*prec*(meanmsmt - meanlen)
    postmeanlen = meanlen + varlen*prec*(meanmsmt - meanlen)
    postvarlen = varlen * (1-varlen*prec)
    postcov = cov * (1-varlen*prec)
    postvarmid = varmid - cov*cov*prec
    return np.array((postmeanmid, postmeanlen)), np.array(((postvarmid,postcov),
                                                           (postcov,postvarlen)))

def _eigs2x2(mtx):
    if abs(mtx[0,1]) < 1e-10:
        return mtx[[0,1],[0,1]], np.eye(2)
    vardiff = (mtx[0,0]-mtx[1,1])/2
    varmean = (mtx[0,0]+mtx[1,1])/2
    dterm = hypot(vardiff, mtx[0,1])
    eigs = np.array((varmean + dterm, varmean - dterm))
    vecs = np.array(((vardiff + dterm, mtx[0,1]),(vardiff - dterm, mtx[0,1])))
    vecs /= np.hypot(vecs[:,0], vecs[:,1])[:,None]
    assert all(eigs > 0)
    if eigs[1] > eigs[0]:
        vecs = vecs[::-1].copy()
        eigs = eigs[::-1].copy()
    return eigs, vecs


if __name__ == '__main__':
    #from matplotlib.axes import add_line
    import matplotlib.pyplot as plt
    
#    minlo, maxlo = 1., 5.
#    minhi, maxhi = 5., 7.
#    minlen, maxlen = .5, 2.5
    minlo, maxlo = 3., 5.
    minhi, maxhi = 5., 7.
    minlen, maxlen = 1.5, 3.5
    
    
    center = (minhi+minlo+maxhi+maxlo)/4
    meanlen = (maxhi+minhi-maxlo-minlo)/4
    
    fig, ax = plt.subplots()
    
    rec = plt.Rectangle(((maxlo+minhi)/2,(minhi-maxlo)/2), (maxhi-minhi)/2**.5,
                         (maxlo-minlo)/2**.5, angle=45., fill=False, edgecolor='r')
    ax.add_patch(rec)
    #plt.show()
    rec = plt.Rectangle((center-5, minlen), 10, maxlen-minlen, angle=0.,
                        fill=False, edgecolor='b')
    ax.add_patch(rec)
    
    pmean, pcov = uvBound2xyNormal(minlo,maxlo,minhi,maxhi,minlen,maxlen)
    pcoveigs, pcovvecs = _eigs2x2(pcov)
    
    angle = np.arctan2(pcovvecs[0,1],pcovvecs[0,0]) * 180/3.14159
    if pcovvecs[0,0] < 0: pcovvecs[0] *= -1
    if pcovvecs[1,1] < 0: pcovvecs[1] *= -1
    pcc = (pcoveigs*12)**.5
    lowerleftx = pmean[0]-pcovvecs[0,0]*pcc[0]/2-pcovvecs[1,0]*pcc[1]/2
    lowerlefty = pmean[1]-pcovvecs[0,1]*pcc[0]/2-pcovvecs[1,1]*pcc[1]/2
    rec = plt.Rectangle((lowerleftx, lowerlefty), pcc[0], pcc[1],
                        angle=angle, fill=False, edgecolor='g')
    ax.add_patch(rec)
    
    plt.axis('equal')
    #plt.show()