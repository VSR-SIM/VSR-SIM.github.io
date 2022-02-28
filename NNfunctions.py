import datetime
import math
import os

import torch
import time

import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import glob

import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
from skimage import exposure

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()


import numpy as np
from PIL import Image

from models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary

	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict

from argparse import Namespace


def GetOptions():
    # training options
    opt = Namespace()
    opt.model = 'rcan'
    opt.n_resgroups = 3
    opt.n_resblocks = 10
    opt.n_feats = 96
    opt.reduction = 16
    opt.narch = 0
    opt.norm = 'minmax'

    opt.cpu = False
    opt.multigpu = False
    opt.undomulti = False
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')

    opt.imageSize = 512
    opt.weights = "model/simrec_simin_gtout_rcan_512_2_ntrain790-final.pth"
    opt.root = "model/0080.jpg"
    opt.out = "model/myout"

    opt.task = 'simin_gtout'
    opt.scale = 1
    opt.nch_in = 9
    opt.nch_out = 1


    return opt


def GetOptions_allRnd_0215():
    # training options
    opt = Namespace()
    opt.model = 'rcan'
    opt.n_resgroups = 3
    opt.n_resblocks = 10
    opt.n_feats = 48
    opt.reduction = 16
    opt.narch = 0
    opt.norm = 'adapthist'

    opt.cpu = False
    opt.multigpu = False
    opt.undomulti = False
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')

    opt.imageSize = 512
    opt.weights = "model/0216_SIMRec_0214_rndAll_rcan_continued.pth"
    opt.root = "model/0080.jpg"
    opt.out = "model/myout"

    opt.task = 'simin_gtout'
    opt.scale = 1
    opt.nch_in = 9
    opt.nch_out = 1


    return opt



def GetOptions_allRnd_0317():
    # training options
    opt = Namespace()
    opt.model = 'swin3d'
    opt.norm = 'minmax'

    opt.cpu = False
    opt.multigpu = False
    opt.undomulti = False
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')

    opt.imageSize = 512
    opt.weights = "model/DIV2K_randomised_3x3_20200317.pth"
    opt.root = "model/0080.jpg"
    opt.out = "model/myout"

    opt.task = 'simin_gtout'
    opt.scale = 1
    opt.nch_in = 9
    opt.nch_out = 1


    return opt


def GetOptions_Swin_2702():

    # training options
    opt = Namespace()
    opt.model = 'rcan'
    opt.n_resgroups = 3
    opt.n_resblocks = 10
    opt.n_feats = 96
    opt.reduction = 16
    opt.narch = 0
    opt.norm = 'minmax'

    opt.cpu = False
    opt.multigpu = False
    opt.undomulti = False
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')

    opt.imageSize = 512
    opt.weights = "model/DIV2K_randomised_3x3_20200317.pth"
    opt.root = "model/0080.jpg"
    opt.out = "model/myout"

    opt.task = 'simin_gtout'
    opt.scale = 1
    opt.nch_in = 9
    opt.nch_out = 1


    return opt


def LoadModel(opt):
    print('Loading model')
    print(opt)

    net = GetModel(opt)
    print('loading checkpoint',opt.weights)
    checkpoint = torch.load(opt.weights,map_location=opt.device)

    if type(checkpoint) is dict:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if opt.undomulti:
        state_dict = remove_dataparallel_wrapper(state_dict)
    net.load_state_dict(state_dict)

    return net


def prepimg(stack,self):

    inputimg = stack[:9]

    if self.nch_in == 6:
        inputimg = inputimg[[0,1,3,4,6,7]]
    elif self.nch_in == 3:
        inputimg = inputimg[[0,4,8]]

    if inputimg.shape[1] > 512 or inputimg.shape[2] > 512:
        print('Over 512x512! Cropping')
        inputimg = inputimg[:,:512,:512]


    if self.norm == 'convert': # raw img from microscope, needs normalisation and correct frame ordering
        print('Raw input assumed - converting')
        # NCHW
        # I = np.zeros((9,opt.imageSize,opt.imageSize),dtype='uint16')

        # for t in range(9):
        #     frame = inputimg[t]
        #     frame = 120 / np.max(frame) * frame
        #     frame = np.rot90(np.rot90(np.rot90(frame)))
        #     I[t,:,:] = frame
        # inputimg = I

        inputimg = np.rot90(inputimg,axes=(1,2))
        inputimg = inputimg[[6,7,8,3,4,5,0,1,2]] # could also do [8,7,6,5,4,3,2,1,0]
        for i in range(len(inputimg)):
            inputimg[i] = 100 / np.max(inputimg[i]) * inputimg[i]
    elif 'convert' in self.norm:
        fac = float(self.norm[7:])
        inputimg = np.rot90(inputimg,axes=(1,2))
        inputimg = inputimg[[6,7,8,3,4,5,0,1,2]] # could also do [8,7,6,5,4,3,2,1,0]
        for i in range(len(inputimg)):
            inputimg[i] = fac * 255 / np.max(inputimg[i]) * inputimg[i]


    inputimg = inputimg.astype('float') / np.max(inputimg) # used to be /255
    widefield = np.mean(inputimg,0)

    if self.norm == 'adapthist':
        for i in range(len(inputimg)):
            inputimg[i] = exposure.equalize_adapthist(inputimg[i],clip_limit=0.001)
        widefield = exposure.equalize_adapthist(widefield,clip_limit=0.001)
    else:
        # normalise
        inputimg = torch.tensor(inputimg).float()
        widefield = torch.tensor(widefield).float()
        widefield = (widefield - torch.min(widefield)) / (torch.max(widefield) - torch.min(widefield))

        if self.norm == 'minmax':
            for i in range(len(inputimg)):
                inputimg[i] = (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))
        elif 'minmax' in self.norm:
            fac = float(self.norm[6:])
            for i in range(len(inputimg)):
                inputimg[i] = fac * (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))



    # otf = torch.tensor(otf.astype('float') / np.max(otf)).unsqueeze(0).float()
    # gt = torch.tensor(gt.astype('float') / 255).unsqueeze(0).float()
    # simimg = torch.tensor(simimg.astype('float') / 255).unsqueeze(0).float()
    # widefield = torch.mean(inputimg,0).unsqueeze(0)


    # normalise
    # gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
    # simimg = (simimg - torch.min(simimg)) / (torch.max(simimg) - torch.min(simimg))
    # widefield = (widefield - torch.min(widefield)) / (torch.max(widefield) - torch.min(widefield))
    inputimg = torch.tensor(inputimg).float()
    widefield = torch.tensor(widefield).float()
    return inputimg,widefield

def save_image(data, filename,cmap):
    sizes = np.shape(data)
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap)
    plt.savefig(filename, dpi = sizes[0])
    plt.close()


def EvaluateModel(net,opt,stack,outfile):

    os.makedirs(opt.out, exist_ok=True)

    print(stack.shape)
    inputimg, widefield = prepimg(stack, opt)

    if opt.norm == 'convert' or 'minmax' in opt.norm or 'adapthist' in opt.norm:
        cmap = 'magma'
    else:
        cmap = 'gray'

    # skimage.io.imsave('%s_wf.png' % outfile,(255*widefield.numpy()).astype('uint8'))
    wf = (255*widefield.numpy()).astype('uint8')
    wf_upscaled = skimage.transform.rescale(wf,1.5,order=3,multichannel=False) # should ideally be done by drawing on client side, in javascript
    save_image(wf_upscaled,'%s_wf.png' % outfile,cmap)

    # skimage.io.imsave('%s.tif' % outfile, inputimg.numpy())

    inputimg = inputimg.unsqueeze(0)

    with torch.no_grad():
        if opt.cpu:
            sr = net(inputimg)
        else:
            sr = net(inputimg.cuda())
        sr = sr.cpu()
        sr = torch.clamp(sr,min=0,max=1)
        print('min max',inputimg.min(),inputimg.max())

        pil_sr_img = toPIL(sr[0])

    if opt.norm == 'convert':
        pil_sr_img = transforms.functional.rotate(pil_sr_img,-90)

    #pil_sr_img.save('%s.png' % outfile) # true output for downloading, no LUT
    sr_img = np.array(pil_sr_img)
    sr_img = exposure.equalize_adapthist(sr_img,clip_limit=0.01)
    skimage.io.imsave('%s.png' % outfile, sr_img) # true out for downloading, no LUT

    sr_img = skimage.transform.rescale(sr_img,1.5,order=3,multichannel=False) # should ideally be done by drawing on client side, in javascript
    save_image(sr_img,'%s_sr.png' % outfile,cmap)
    return outfile + '_sr.png', outfile + '_wf.png', outfile + '.png'
