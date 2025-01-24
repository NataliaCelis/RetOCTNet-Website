IMAGE_DIR="./TestOCT/"
SAVE_DIR="./TestOCT"
NUM_IMAGES=None
POST_PROCESSING=False
MOUSE=False
DISCARD=False
DISCARD_POST=512



python execute.py --image_dir "$IMAGE_DIR"  --save_dir "$SAVE_DIR"  --post_processing $POST_PROCESSING --mouse $MOUSE --discard $DISCARD  --discard_post $DISCARD_POST


#Resize images when saving as tiffs first 