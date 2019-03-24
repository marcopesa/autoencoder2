from utils import coco
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#Create dataClass
myCocoDataClass = coco.CocoImagesDataClass()

# Download coco dataset
myCocoDataClass.maybe_download_and_extract_coco()

# Load records
myCocoDataClass.load_records(trainSet=True)
myCocoDataClass.load_records(trainSet=False)

# Generate vocabulary
myCocoDataClass.generate_vocabulary()

# Download vgg16 weights
myCocoDataClass.maybe_download_and_extract_vgg16weights()

myCocoDataClass.produceVgg16Fc7()







