
import tensorflow as tf
from tensorflow import keras
import numpy as np 
from datagen import * 
from keras.utils import load_img,img_to_array
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import os 

from PIL import Image

import glob



def drn_execute(files, discard=False, pad=12):
    '''
    Get the predictions of drn for a list of files (list must include directory)
    '''
    path_to_weights='./weights.h5'
    model = tf.keras.models.load_model(path_to_weights, compile=False)

    #Get images 
    ims, pad_h, pad_w=get_images2(files) 

    if len(files)==1:
        ims = np.repeat(ims, 2, axis=-1)

   # print(f' Images shape: {ims.shape}')

    #print(f' pad_h: {pad_h}')
   # print(f' pad_w : {pad_w}')

    patch, npatch_h, npatch_v =make_patches2(ims)
    
    #print(f' Patches shape: {patch.shape}')
    #print(f' npatch_h : {npatch_h}')
    #print(f' npatch_v : {npatch_v}')
    #Patches and normalize
    patches=normalize(patch)

    if discard:
        retain_indices = np.arange(6, patches.shape[3], 8)
        retain_indices = np.sort(np.concatenate([retain_indices, retain_indices + 1]))
        patches = patches[:, :, :, retain_indices] 
        patch = patch[:, :, :, retain_indices] 


    #Predict
    preds = model.predict(dataGen(1,patches))  
    hard=np.argmax(preds, axis=-1).transpose(1,2,0)
    hard_predicions=np.expand_dims(hard, axis=2)

    #print(f' hard_predicions: {hard_predicions.shape}')

    #ims=reconstruct_images(np.zeros((2048,1024,1,2)), patch, 4, 2,0,24)

    if discard:
        #print(f'patch :{patch.shape}')
       #print(f'pad_w :{pad_w}')

        stitched_images,stitched_predictions, _=reconstruct_after_discard(patch, preds, pad_w )
        stitched_predictions=np.expand_dims(stitched_predictions, axis=2)
       # stitched_predictions=np.expand_dims(stitched_predictions[:,:,:,0], axis=-1)
    else:
        stitched_images=reconstruct_images(ims, patch, npatch_v, npatch_h,pad_h,pad_w)

        stitched_predictions=reconstruct_images(ims,hard_predicions, npatch_v, npatch_h,pad_h,pad_w)

   # print(f'after :{stitched_predictions.shape}')

    if len(files)==1: 
        print(stitched_predictions.shape)
        stitched_predictions=np.expand_dims(stitched_predictions[:,:,:,0], axis=-1)
        print(f'after :{stitched_predictions.shape}')

  #  ims_full,preds_hard_full, preds_soft_full=reconstruct2(ims,patches, preds, pad_h, pad_w, npatch_h , npatch_v, patch_size=512)

    return stitched_images,stitched_predictions #, preds_soft_full , patch 

def get_images2(scan_list):
    #preallocate values in np arrays 

    #Calculate patching with one image: 
    h,w,_=img_to_array(load_img(scan_list[0],color_mode='grayscale')).shape

   # print(f'h:{h}')
   # print(f'w:{w}')

    pad_h = (512 - (h % 512)) % 512
    pad_w = (512 - (w % 512)) % 512

   # print(f'pad_h:{pad_h}')
 #   print(f'pad_w:{pad_w}')

    x=np.zeros((h+pad_h,w+pad_w, 1, len(scan_list)))
    for i in range(len(scan_list)):
        IM=img_to_array(load_img(scan_list[i],color_mode='grayscale'))
        
        x[:,:,:,i]=np.pad(IM,((pad_h,0),(pad_w,0),(0,0)), mode='symmetric')

    return x, pad_h, pad_w


def make_patches2(x, hsize=512):

    npatch_h=x.shape[1]//hsize #Number of patches per image in horizontal d
    npatch_v=x.shape[0]//hsize #(x.shape[0]//hsize) #Number of patches per image in vertical d by 2 bc we discard lower part

    
    idx_h=np.arange(0,npatch_h)*hsize
    idx_v=np.arange(0,npatch_v)*hsize
    patch_x=np.zeros((hsize,hsize,x.shape[2],x.shape[3]*npatch_h*npatch_v))
    c=-1
    for i in range(x.shape[3]):
        im=x[:,:,:,i]
        for j in range(npatch_v):
            for k in range(npatch_h):
                c+=1                
                patch_x[:,:,:,c]=im[idx_v[j]:idx_v[j]+hsize,idx_h[k]:idx_h[k]+hsize,:]
                
    return patch_x , npatch_h, npatch_v #[:-512,:,:,:]   

def reconstruct_images(original_images, patches, npatch_v, npatch_h,vpad,hpad):

    _,_,_, n= original_images.shape
    reconstruct=np.zeros((original_images.shape))
  #  print(f'Recontruct shape:{reconstruct.shape}')
   # print(f'patches shape:{patches.shape}')
    p=0
    for k in range(n):
       # print(f'k:{k}')
        for i in range(npatch_v):
          #  print(f'i:{i}')
            for j in range (npatch_h):
                reconstruct[i*512:i*512+512,j*512:j*512+512,:,k]= patches[:,:,:,p]
                p+=1

    return reconstruct[vpad:,hpad:,:,:]

#Form images
def reconstruct_after_discard(x,pred,pad=12):
    x_full=np.zeros((x.shape[0],x.shape[1]*2,x.shape[2],x.shape[3]//2))
    pred_train_hard = np.argmax(pred, axis=-1)

    x_full_predictions_hard=np.zeros((x.shape[0],x.shape[1]*2,x.shape[3]//2))
    x_full_predictions_soft=np.zeros((x.shape[0],x.shape[1]*2,3,x.shape[3]//2))


    print(f'x full prediciton shape: {x_full_predictions_hard.shape}')

    idx=-1
    for i in range(int(x.shape[3])):
        #print(i)
        if i%2==0:
            idx+=1
            #print('here')
            #print(idx)
            x_full[:,:-512,:,idx]=x[:,:,:,i]
            #print(np.sum(y_train_patch[:,:,i]))
            x_full_predictions_hard[:,:-512,idx]=pred_train_hard[i,:,:]
            x_full_predictions_soft[:,:-512,:,idx]=pred[i,:,:,:]
        else:
            #print('not here')
            #print(idx)
            x_full[:,512:,:,idx]=x[:,:,:,i]
            #print(np.sum(y_train_patch[:,:,i]))
            x_full_predictions_hard[:,512:,idx]=pred_train_hard[i,:,:]
            x_full_predictions_soft[:,512:,:,idx]=pred[i,:,:,:]
    
    #Get rid of the padding!
    return x_full[:,pad:,:],x_full_predictions_hard[:,pad:,:], x_full_predictions_soft[:,pad:,:,:] #patching done on one side only 

def normalize(patches):
    scale=StandardScaler()
    patch_squeeze=patches.squeeze(axis=2)
    patch_squeeze = np.reshape(patch_squeeze, (patch_squeeze.shape[2],-1)) # (n_samples, n_features)
    scale.fit(patch_squeeze)
      
   
    patches_norm=scale.transform(patch_squeeze)
    patches_train_norm = np.reshape(patches_norm, (patches.shape[0], patches.shape[1], patches_norm.shape[0]))[:,:, np.newaxis, :]

    return patches_train_norm


def plot_prediction(img,pred,idx,map='gray'):
    '''
    When there are no GTs. 
    '''
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    
    #id_ex = np.random.randint(0,img.shape[3])
    im = img[:,:,0,idx]
    im_pred= pred[:,:,0,idx]
    
    labels = np.unique(im_pred)

    axes[0].axis("off") 
    axes[0].set_title('OCT Image') #+str(idx[n]))
    axes[0].imshow(im, cmap='gray')

    axes[1].axis("off") 
    axes[1].set_title(f'Prediction Map')
    axes[1].imshow(im_pred, cmap=map)

    #plt.subplots_adjust(wspace=.1, hspace=.01)
    #plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.tight_layout()
    #plt.savefig(f'Image_{name}_{idx}.png',bbox_inches='tight')
    plt.show()
    

    #plt.tight_layout()  


def get_filename(path):
    # Get the base name of the file (e.g., 'image_3484825-2_OD_A_1x1_0_RegAvg0000134_Scan_1.tiff')
    basename = os.path.basename(path)
    
    # Split the basename into the name and extension parts
    filename, extension = os.path.splitext(basename)
    
    return filename


def save_predictions(predictions, list_of_files, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plist = [predictions[:, :, :, i] for i in range(predictions.shape[3])]

    filenames=[get_filename(i) for i in list_of_files]

    a=zip(filenames, plist)
    for name, prediction in a:
        
        Image.fromarray(prediction.astype(np.uint8).squeeze()).save(os.path.join(save_path, name+'_DRN.tiff'))

def save_images_with_color(predictions, list_of_files, save_path, ext='DRN'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Define a colormap: you can customize this as needed
    colormap = {
        0: [0, 0, 0],       # black for 0
        1: [128, 128, 128], # grey for 1
        2: [255, 255, 255], # white for 2
    }

    plist = [predictions[:, :, :, i] for i in range(predictions.shape[3])]
    filenames = [get_filename(i) for i in list_of_files]



    for name, prediction in zip(filenames, plist):
        # Ensure the prediction array has the right shape
        prediction = np.squeeze(prediction)
        
        # Create an RGB image using the colormap
        height, width = prediction.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for value, color in colormap.items():
            rgb_image[prediction == value] = color
        
        # Convert the array to an image and save it
        image = Image.fromarray(rgb_image)
        image.save(os.path.join(save_path, name +ext+ '.tiff'))

def save_images(predictions, list_of_files, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plist = [predictions[:, :, :, i] for i in range(predictions.shape[3])]

    filenames=[get_filename(i) for i in list_of_files]

    a=zip(filenames, plist)
    for name, prediction in a:
        prediction = (prediction / prediction.max() * 255).astype(np.uint8)
        
        Image.fromarray(prediction.squeeze()).save(os.path.join(save_path, name+'.tiff'))



def load_tiff_images(folder_path):
    # Find all .tiff files in the specified folder
    tiff_files = glob.glob(os.path.join(folder_path, '*.tiff'))



    # List to store the loaded images
    images = []

    # Load each image and convert it to a NumPy array
    for file in tiff_files:
        img = Image.open(file)
        img_array = np.array(img)
        images.append(img_array)

    # Convert the list of images to a NumPy array
    if images:
        images_array = np.stack(images, axis=2)
    else:
        images_array = np.array([])

    return images_array


import re
#https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def flatten(l):
    return [item for sublist in l for item in sublist]