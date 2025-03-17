import os,time,scipy.io

import numpy as np
import rawpy
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim

# from model import SeeInDark
# from skipnet_flops_cnn2 import SkipNet as SkipNet
from gfnet_v1 import GFNet as GFNet
# from net_umgf import UMGFNet as SkipNet
from toimage import toimage

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
model_nm = 'sony_gfnet_v1'
epoch  = '6800'
model = GFNet(ch=64)

input_dir = '/home/ubuntu/yxxxl/oridata/Sony/Sony/short/'
gt_dir = '/home/ubuntu/yxxxl/oridata/Sony/Sony/long/'
m_path = '/home/ubuntu/yxxxl/GDSR/See_In_the_Dark/saved_model_' + model_nm + '/'
m_name = 'checkpoint_sony_e' + epoch + '.pth'
result_dir = '/home/ubuntu/yxxxl/GDSR/See_In_the_Dark/result'+'_' + model_nm + '/'

# 
test_input_paths = glob.glob('/home/ubuntu/yxxxl/oridata/Sony/Sony/short/1*_00_0.1s.ARW')
# test_input_paths = glob.glob('/home/ubuntu/yxxxl/oridata/Sony/Sony/short/10016_00_0.1s.ARW')
# test_input_paths = glob.glob('/home/notebook/data/group/lxy/Flash/data/SID/Sony/short/')
test_gt_paths = []
# nm=0
for x in test_input_paths:
    test_gt_paths += glob.glob('/home/ubuntu/yxxxl/oridata/Sony/Sony/long/*' + x[-17:-12] + '*.ARW')
# print(nm)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
#get test IDs
# test_fns = glob.glob(gt_dir + '*.ARW')

test_ids = []
for i in range(len(test_gt_paths)):
    _, test_fn = os.path.split(test_gt_paths[i])
    test_ids.append(int(test_fn[0:5]))



def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = np.maximum(raw - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    # out = im

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def pack_raw_bi(im):
    #pack Bayer image to 4 channels
    # im = raw.raw_image_visible.astype(np.float32) 
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    # im = cv2.bilateralFilter(im, 10, 30, 30)
    # im = cv2.GaussianBlur(im, (51,51),30)
    
    im = np.clip(im, 0, 1)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    im = np.expand_dims(im,axis=2) 
    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    out[:,:,0] = cv2.bilateralFilter(out[:,:,0], 10, 30, 30)[:,:,None]
    out[:,:,1] = cv2.bilateralFilter(out[:,:,1], 10, 30, 30)[:,:,None]
    out[:,:,2] = cv2.bilateralFilter(out[:,:,2], 10, 30, 30)[:,:,None]
    out[:,:,3] = cv2.bilateralFilter(out[:,:,3], 10, 30, 30)[:,:,None]
    # print(out.shape)
    return out[:,:,:,0] 

def depack(im):
    H,W,_ = im.shape
    H = H*2
    W = W*2
    out = np.zeros((H, W))
    out[0:H:2,0:W:2] = im[:,:,0]
    out[0:H:2,1:W:2] = im[:,:,1]
    out[1:H:2,1:W:2] = im[:,:,2]
    out[1:H:2,0:W:2] = im[:,:,3]
    return out
    
    
model_dict = torch.load( m_path + m_name ,map_location=device)
model.load_state_dict(model_dict)
model = model.to(device)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
with torch.no_grad():
    for test_id in test_ids:
        print('test_id', test_id)
        #test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_00_0.1s.ARW'%test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            _, in_fn = os.path.split(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%test_id) 
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            in_exposure =  float(in_fn[9:-5])
            gt_exposure =  float(gt_fn[9:-5])
            ratio = min(gt_exposure/in_exposure,300)

            raw = rawpy.imread(in_path)
            im = raw.raw_image_visible.astype(np.float32) 
            input_full = np.expand_dims(pack_raw(im),axis=0) *ratio
            input_bi =  np.expand_dims(pack_raw_bi(im),axis=0) *ratio
            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)	

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full = np.expand_dims(np.float32(im/65535.0),axis = 0)

            input_full = np.minimum(input_full,1.0)
            input_bi = np.clip(input_bi, 0, 1)
            in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
            in_bi = torch.from_numpy(input_bi).permute(0,3,1,2).to(device)
            in_img = torch.cat((in_img, in_bi), dim=1)
            print(in_img.shape)
            # torch.cuda.synchronize()
            # start = time.time()
            out_img = model(in_img)
            # torch.cuda.synchronize()
            # end = time.time()
            # print(end-start)
            inp_img = in_bi.permute(0, 2, 3, 1).cpu().data.numpy()
            inp_img = depack(inp_img[0])
            inp_img = inp_img  * (16383-512)
            inp_img = np.clip(inp_img, 0, 16383)
            
            # gt_raw = rawpy.imread(gt_path)
            # gt_raw.raw_image_visible[:] = inp_img.astype(int)
            # unet = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()

            output = np.minimum(np.maximum(output,0),1)

            output = output[0,:,:,:]
            gt_full = gt_full[0,:,:,:]
            scale_full = scale_full[0,:,:,:]
            origin_full = scale_full
            scale_full = scale_full*np.mean(gt_full)/np.mean(scale_full) # scale the low-light image to the same mean of the groundtruth
            
            # toimage(origin_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_ori.png'%(test_id,ratio))
            toimage(output*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '{}_{}.png'.format(test_id,epoch))
            # toimage(unet*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '{}_{}_bi.png'.format(test_id,epoch))
            # toimage(scale_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_scale.png'%(test_id,ratio))
            # toimage(gt_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_gt.png'%(test_id,ratio))


