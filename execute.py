import tensorflow as tf
from usage_utils import *
import glob
import os
import argparse

from post_processing import *

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    parser = argparse.ArgumentParser(description="Train Unet for displacement prediction.")

    parser.add_argument('--image_dir', type=str,required=False, help='List of images with parentdir to use for training.')

    parser.add_argument('--save_dir', type=str, default=True, required=True, help='Save images in a folder')

    parser.add_argument('--num_images', type=int,required=False, help='Cap images if too many.')

    parser.add_argument('--post_processing', type=str2bool, default=False, required=False, help='Choose if post_processing is applied.')

    parser.add_argument('--discard', type=str2bool, default=False, required=False, help='Discard the frist six patches (Built for mouse data analysis).')

    parser.add_argument('--mouse', type=str2bool, default=False, required=False, help='Choose if post_processing is applied.')

    parser.add_argument('--discard_post', type=int, default=False, required=False, help='Discarding after segmentation, length wise. No image processing involved.')

    args = parser.parse_args()

    
    if args.num_images: 
        list_of_files=sorted(filter(os.path.isfile,glob.glob(args.image_dir + '/**/*', recursive=True))) [:args.num_images]
        # Get the list of files, sorted and filtered by the specified criteria
        #print(list_of_files)
    else:
        list_of_files=sorted(filter(os.path.isfile,glob.glob(args.image_dir + '/**/*', recursive=True))) 
        #print('Here')
        #print(list_of_files)

        if args.mouse:

            list_of_files = [file for file in list_of_files1 if file.endswith('Scan_2.tiff')]

    tiff_files = [file for file in list_of_files if file.lower().endswith(('.tiff', '.tif'))]

    print(f'Retrieved {len(tiff_files)} images. Analyzing...')

    images,predictions=drn_execute(tiff_files, discard=args.discard)
    
    if args.discard_post:
        images=images[:args.discard_post]
        predictions=predictions[:args.discard_post]


    if args.post_processing:
        print('Post Processing Images ..............................')
        predictions = process_images(predictions)
        ext='_DRN_post_processed'
    else:

        ext='_DRN'

    save_images_with_color(predictions, tiff_files, args.save_dir+'/masks', ext=ext)

    #save_images(images, tiff_files, args.save_dir+'/images')

    

    


if __name__ == "__main__":
    main()
