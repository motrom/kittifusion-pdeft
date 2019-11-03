#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 2/15/19
basic information on different scenes for kitti tracking benchmark
"""
import os

nfiles_training = [154, 447, 233, 144, 314, 297, 270, 800, 390, 803, 294,
                   373, 78, 340, 106, 376, 209, 145, 339, 1059, 837]
startfiles_training = [0,181,0,0,0,0,0,0,0,0]

calib_map_training = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3]

actual_nfiles = [154, 447, 233, 144, 314, 297, 270, 800, 390, 803, 294,
                   373, 78, 340, 106, 376, 209, 145, 339, 1059, 837]

sceneranges = [(0,153), (181,447), (0,233), (0,144), (0,314), (0, 296), (0,270),
               (0,799), (0,390), (0,802)]

# makes different folders for each scene, when making output folders
def makeOutputFolder(loc='../tracking/measurements', name='a', number=20):
    nloc = loc + '/' + name
    files = os.listdir(loc)
    if name not in files:
        os.mkdir(nloc)
    files = os.listdir(nloc)
    for i in range(number):
        stri = '{:04d}'.format(i)
        if stri not in files:
            os.mkdir(nloc + '/' + stri)
            
        

        
def convertTrackGt2DetectGt(infolder, outfolder, numscenes=21):
    outputformat = outfolder+'/{:06d}.txt'
    nfiles_cum = [sum(nfiles_training[:scene]) for scene in range(numscenes)]
    output = []
    for scene in range(numscenes):
        infile = infolder+'/{:04d}.txt'.format(scene)
        with open(infile, 'r') as infd: instr = infd.read().split('\n')
        scenestart = nfiles_cum[scene]
        filenum = 0
        for line in instr:
            if line=='': continue
            linesplit = line.split(' ')
            linefilenum = int(linesplit[0])
            while linefilenum > filenum:
                # save current output
                # while instead of if, because you might skip some times
                # need to write blank files for those times
                with open(outputformat.format(scenestart+filenum), 'w') as outfd:
                    outfd.write('\n'.join(output))
                output = []
                filenum += 1
            line = ' '.join(linesplit[2:]) # remove frameid and trackid
            output.append(line)
        scenelen = nfiles_training[scene]
        while scenelen > filenum:
            with open(outputformat.format(scenestart+filenum), 'w') as outfd:
                outfd.write('\n'.join(output))
            output = []
            filenum += 1