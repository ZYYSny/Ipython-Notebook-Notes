# Ipython-Notebook-Notes
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, filter
import sys, time, argparse, pylab, operator, csv
import util
import os
import urllib
import cv2
import json
import scipy.io as sio
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels,create_pairwise_bilateral
from pydensecrf.utils import unary_from_softmax,create_pairwise_gaussian


# COCO API
coco_root = '/home/wfge/database/MsCoco'  # modify to point to your COCO installation
sys.path.insert(0, coco_root + '/PythonAPI')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask

# CAFFE
#caffe_root = '/home/wfge/Deeplearning-Caffe/Caffe-ExcitationBP'
caffe_root = '../'
superpixel_root = '/home/wfge/database/MsCoco/edges/train2014/'
saliency_root = '/home/wfge/database/MsCoco/segments/train2014/'
sys.path.insert(0, caffe_root + '/python')
import caffe

# load COCO train2014
imset   = 'train2014'
imgDir  = '%s/images/%s/'%(coco_root, imset)
annFile = '%s/annotations/instances_%s.json'%(coco_root, imset)
cocoAnn = COCO(annFile)
cocoAnn.info()
catIds  = cocoAnn.getCatIds()
catList = cocoAnn.loadCats(catIds)
imgIds  = cocoAnn.getImgIds()

# PARAMS
tags, tag2ID = util.loadTags(caffe_root + '/models/COCO/catName.txt')
imgScale = 224
topBlobName = 'loss3/classifier'
topLayerName = 'loss3/classifier'
secondTopLayerName = 'pool5/7x7_s1'
secondTopBlobName = 'pool5/7x7_s1'
outputLayerName = 'pool2/3x3_s2'
outputBlobName = 'pool2/3x3_s2'

caffe.set_mode_gpu()
caffe.set_device(3)
att_net = caffe.Net(caffe_root+'/models/COCO/deploy.prototxt',
                caffe_root+'/models/COCO/GoogleNetCOCO.caffemodel',
                caffe.TRAIN)
#net = caffe.Net('../models/COCO/deploy.prototxt',
#                '../models/COCO/GoogleNetCOCO.caffemodel',
#                caffe.TRAIN)


# read proposals from text
candidate_rois = {}
with open('/home/wfge/Deeplearning-Caffe/EdgeDetection/edges/mscoco_train2014_candidate_rois.txt') as f:
    for line in f:
        temp_lines = line.split(' ')
        img_name = temp_lines[0]+'.jpg'
        img_label = int(temp_lines[1])
        img_cat = tags[img_label-1]
        bbs_x = int(temp_lines[2])
        bbs_y = int(temp_lines[3])
        bbs_w = int(temp_lines[4])
        bbs_h = int(temp_lines[5])
        candidate_rois[img_name] = {}
        candidate_rois[img_name][img_cat] = []		

with open('/home/wfge/Deeplearning-Caffe/EdgeDetection/edges/mscoco_train2014_candidate_rois.txt') as f:
    for line in f:
        temp_lines = line.split(' ')
        img_name = temp_lines[0]+'.jpg'
        img_label = int(temp_lines[1])
        img_cat = tags[img_label-1]
        bbs_x = int(temp_lines[2])
        bbs_y = int(temp_lines[3])
        bbs_w = int(temp_lines[4])
        bbs_h = int(temp_lines[5])
        candidate_rois[img_name][img_cat].append([bbs_x,bbs_y,bbs_w,bbs_h])
		
def doExcitationBackprop(net, img, tagNames):
    # load image, rescale
    minDim = min(img.shape[:2])
    newSize = (int(img.shape[0]*imgScale/float(minDim)), int(img.shape[1]*imgScale/float(minDim)))
    imgS = transform.resize(img, newSize)

    # reshape net
    net.blobs['data'].reshape(1,3,newSize[0],newSize[1])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    # forward pass
    net.blobs['data'].data[...] = transformer.preprocess('data', imgS)
    out = net.forward(end = topLayerName)
 
    scores = net.blobs[topBlobName].data[0].reshape((len(tags),-1)).max(1).flatten() # pre-softmax scores
    tagScore = util.getTagScore(scores, tags, tag2ID)
    tagScore.sort(key = operator.itemgetter(1), reverse = True)
    #print(tagScore[:10])
    
    # switch to the excitation backprop mode
    caffe.set_mode_eb_gpu() 
    attMaps = []
    for tagName in tagNames:
        tagID = tag2ID[tagName]
        net.blobs[topBlobName].diff[0][...] = 0
        net.blobs[topBlobName].diff[0][tagID] = np.exp(net.blobs[topBlobName].data[0][tagID].copy())
        net.blobs[topBlobName].diff[0][tagID] /= net.blobs[topBlobName].diff[0][tagID].sum()

        # invert the top layer weights
        net.params[topLayerName][0].data[...] *= -1
        out = net.backward(start = topLayerName, end = secondTopLayerName)
        buff = net.blobs[secondTopBlobName].diff.copy()
    
        # invert back
        net.params[topLayerName][0].data[...] *= -1 
        out = net.backward(start = topLayerName, end = secondTopLayerName)
    
        # compute the contrastive signal
        net.blobs[secondTopBlobName].diff[...] -= buff
    
        # get attention map
        out = net.backward(start = secondTopLayerName, end = outputLayerName)
        attMap = np.maximum(net.blobs[outputBlobName].diff[0].sum(0), 0)
    
        # resize back to original image size
        attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'nearest')
        attMaps.append(attMap)
    return attMaps

def proposalAttention(net, img, proposals, tagNames, attMaps)
    # attention map normalization
    for i in range(len(tagName)):
        attMap = attMaps[i].copy()
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()
        attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'nearest')
        if 1:
            attMap = filter.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
            attMap -= attMap.min()
            attMap /= attMap.max()
        attMaps[i] = attMap
	
    for category in proposals:
        for rois in category:
            roi_img = img[rois[1]:rois[3],rois[0]:rois[2],:]
            # load image, rescale
            newSize = (224, 224)
            imgS = transform.resize(roi_img, newSize)
            # reshape net
            net.blobs['data'].reshape(1,3,newSize[0],newSize[1])
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
            transformer.set_transpose('data', (2,0,1))
            transformer.set_channel_swap('data', (2,1,0))
            transformer.set_raw_scale('data', 255.0)

            # forward pass
            net.blobs['data'].data[...] = transformer.preprocess('data', imgS)
            out = net.forward(end = topLayerName)
 
            scores = net.blobs[topBlobName].data[0].reshape((len(tags),-1)).max(1).flatten() # pre-softmax scores
            tagScore = util.getTagScore(scores, tags, tag2ID)
            tagScore.sort(key = operator.itemgetter(1), reverse = True)
			
            # switch to the excitation backprop mode
            caffe.set_mode_eb_gpu()
            cat_local_idx = 0			
            for tagName in tagNames:
                if tagName in tagScore[:len(tagNames)]:
                    tagID = tag2ID[tagName]
                    net.blobs[topBlobName].diff[0][...] = 0
                    net.blobs[topBlobName].diff[0][tagID] = np.exp(net.blobs[topBlobName].data[0][tagID].copy())
                    net.blobs[topBlobName].diff[0][tagID] /= net.blobs[topBlobName].diff[0][tagID].sum()

                    # invert the top layer weights
                    net.params[topLayerName][0].data[...] *= -1
                    out = net.backward(start = topLayerName, end = secondTopLayerName)
                    buff = net.blobs[secondTopBlobName].diff.copy()
    
                    # invert back
                    net.params[topLayerName][0].data[...] *= -1 
                    out = net.backward(start = topLayerName, end = secondTopLayerName)
    
                    # compute the contrastive signal
                    net.blobs[secondTopBlobName].diff[...] -= buff
    
                    # get attention map
                    out = net.backward(start = secondTopLayerName, end = outputLayerName)
                    attMap = np.maximum(net.blobs[outputBlobName].diff[0].sum(0), 0)
    
                    # resize back to original image size
                    attMap = transform.resize(attMap, (roi_img.shape[:2]), order = 3, mode = 'nearest')
					# blur the attention map
                    attMap -= attMap.min()
                    if attMap.max() > 0:
                        attMap /= attMap.max()
                    if 1:
                        attMap = filter.gaussian_filter(attMap, 0.02*max(roi_img.shape[:2]))
                        attMap -= attMap.min()
                        attMap /= attMap.max()
                    temp_att_map = attMaps[cat_local_idx].copy()
                    alpha = 1
                    if tagName != category:
                        alpha = 0.5
                    temp_att_map[rois[1]:rois[3],rois[0]:rois[2]] = temp_att_map[rois[1]:rois[3],rois[0]:rois[2]] + alpha*attMap
                cat_local_idx = cat_local_idx + 1 
    return attMaps
	
#imgIdss = [393221]
proposals = {}
#for imgId in imgIdss: 
imgIdx = 0
for imgId in imgIds:    
    # for every category perform excitation bp
    if imgIdx < 82081:
        imgIdx = imgIdx + 1
        continue
    else:
        imgIdx = imgIdx + 1
    # load image annotations
    imgList = cocoAnn.loadImgs(ids=imgId)
    for I in imgList:
        imgName = imgDir + I['file_name']
        img     = caffe.io.load_image(imgName)
        # load category
        imgAnnIds = cocoAnn.getAnnIds(imgIds=imgId)
        imgAnns = cocoAnn.loadAnns(ids=imgAnnIds)
        catName = []
        for imgAnn in imgAnns:
            catId   = imgAnn['category_id']
            catAnns = cocoAnn.loadCats(ids=catId)
            for catAnn in catAnns:
                catName.append(catAnn['name']) 
        catName = list(set(catName))
        if len(catName)==0:
            continue;
        print "Process the %d th image %s."%(imgIdx,imgName)
		# do excitation bp
        attMaps = doExcitationBackprop(att_net, img, catName)
		# for every proposal do excitation bp
        attMaps = proposalAttention(att_net, img, candidate_rois[I['file_name']], catName, attMaps)
		# read superpixels
        superpixel_path = superpixel_root + I['file_name'].split('.')[0] + "_sps.mat"
        superpixels = sio.loadmat(superpixel_path)
        superpixels = superpixels['S']
        sp_num = superpixels.max()
        sp_nums = np.zeros(sp_num,dtype=np.float32)
        sp_saliency_score = np.zeros(sp_num,dtype=np.float32)
        sp_attention_score = np.zeros(sp_num,len(catName),dtype=np.float32)
        # read saliency		
        saliency_path = saliency_root + I['file_name']
        salMaps = sio.loadmat(saliency_path)
        salMaps = salMaps['interseg']
        salMap = np.sum(salMaps,axis = 2)
        salMap -= salMap.min()
        if salMap.max() > 0:
            salMap /= salMap.max()
        # caculate the probability of the every superpixel
        for i in range(superpixels.shape(0)):
            for j in range(superpixels.shape(1)):
                sp_idx = superpixels[i][j]
                sp_nums[sp_idx] = sp_nums[sp_idx] + 1
                sp_saliency_score[sp_idx] = sp_saliency_score[sp_idx] + salMap[i][j]
                for k in range(len(catName)):
                    sp_attention_score[sp_idx][k] = sp_attention_score[sp_idx][k] + attMaps[k][i][j]
        sp_lables = sp_attention_score.max(1)
        att_max = np.zeros(len(catName),dtype=np.float32)
        for i in range(len(catName)):
            att_max[i] = attMaps[i].max()
        for i in range(sp_num):
            sp_saliency_score[i] = sp_saliency_score[i]/sp_nums[i]
            sp_attention_score[sp_idx,:] = sp_attention_score[sp_idx,:]/sp_nums[i]
            currrent_label = sp_lables[i]
            if sp_saliency_score[i] > 0.1:
                if sp_attention_score[sp_idx][currrent_label] < att_max[currrent_label]*0.10 :
                    sp_lables[i] = -1
                else : 				
                    sp_lables[i] = tag2ID(catName[currrent_label])
            else:
                if sp_attention_score[sp_idx][currrent_label] > att_max[currrent_label]*0.75 :
                    sp_lables[i] = tag2ID(catName[currrent_label])
                else : 				
                    sp_lables[i] = -1
        '''
		#util.showAttMap(img, attMaps, catName, overlap = True, blur = True)
        normedAttMaps = util.normedAttMap(img, attMaps,catName)
        #util.showAttMap(img, normedAttMaps, catName, overlap = True, blur = True)
        #dense CRF
        denseCrf = dcrf.DenseCRF2D(img.shape[1],img.shape[0],len(catName)+1)
        hardLabel = util.getHardLabel(img, normedAttMaps,len(catName))
        #util.showHardLabel(img, hardLabel, catName, overlap = True, blur = False)
        imgProposals = util.getProposals(img,hardLabel,catName)
        proposals[I['file_name']] = imgProposals
        '''
		'''
        unaryPot = unary_from_labels(hardLabel, len(catName)+1, gt_prob=0.7, zero_unsure = False)
        denseCrf.setUnaryEnergy(unaryPot)
        denseCrf.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                     normalization=dcrf.NORMALIZE_SYMMETRIC)
        denseCrf.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img.astype(np.uint8),
                                      compat=10,
                                      kernel=dcrf.DIAG_KERNEL,
                                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        # Run five inference steps.
        Q = denseCrf.inference(5)
        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)
        crfMap = MAP.reshape((img.shape[:2]))
        util.showCrfMap(img, crfMap, catName, overlap = True, blur = True)
        '''
        '''
        segMaps = []
        for i in range(len(catName)):
            normedAttMap = np.zeros((2,img.shape[:2][0],img.shape[:2][1]),dtype=float)
            normedAttMap[0] = normedAttMaps[i]
            normedAttMap[1] = 1 - normedAttMaps[i]
            unaryPot = unary_from_softmax(normedAttMap)
            denseCrf.setUnaryEnergy(unaryPot)
            denseCrf.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                         normalization=dcrf.NORMALIZE_SYMMETRIC)
            denseCrf.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img.astype(np.uint8),
                                          compat=10,
                                          kernel=dcrf.DIAG_KERNEL,
                                          normalization=dcrf.NORMALIZE_SYMMETRIC)
            # Run five inference steps.
            crfOut = denseCrf.inference(5)
            segProb = np.array(crfOut).reshape((2,img.shape[0],img.shape[1]))
            segMaps.append(segProb[0])
        util.showAttMap(img, segMaps, catName, overlap = True, blur = True)
        unaryPot = unary_from_softmax(normedAttMaps)
        denseCrf.setUnaryEnergy(unaryPot)
        denseCrf.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                     normalization=dcrf.NORMALIZE_SYMMETRIC)
        denseCrf.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img.astype(np.uint8),
                                      compat=10,
                                      kernel=dcrf.DIAG_KERNEL,
                                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        # Run five inference steps.
        Q = denseCrf.inference(5)
        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)
        crfMap = MAP.reshape((img.shape[:2]))
        util.showCrfMap(img, crfMap, catName, overlap = True, blur = True)
        '''
with open('coco_train2014_proposals80282-82783.json','w') as outfile:
    json.dump(proposals, outfile)
