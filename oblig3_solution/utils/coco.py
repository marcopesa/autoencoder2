########################################################################
#
# Functions for downloading the COCO data-set from the internet
# and loading it into memory. This data-set contains images and
# various associated data such as text-captions describing the images.
#
# http://cocodataset.org
#
# Implemented in Python 3.6
#
########################################################################
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2018 by Magnus Erik Hvass Pedersen
#
########################################################################
import json
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import glob
from tqdm import trange

import generateVocabulary
from utils import downloadCoco
from utils import CNN_network

class CocoImagesDataClass():
    def __init__(self):
        self.data_dir = "data/coco/"

        # Sub-directories for the training- and validation-sets.
        self.train_dir = "data/coco/train2017"
        self.val_dir = "data/coco/val2017"

        # coco images and captions paths
        self.imgSize = [224, 224]
        self.trainBatchSize = 64
        self.valBatchSize  = 64

        # A list holding the image name and the captions
        self.train_records_list = []
        self.val_records_list  = []

        # A iterator counter used while looping through the images
        self.trainImgIter      = 0
        self.trainNumbOfImages = 0

        self.valImgIter      = 0
        self.valNumbOfImages = 0

        self.train_records_list = []
        self.val_records_list  = []

        return

    def set_data_dir(self, new_data_dir):
        """
        Set the base-directory for data-files and then
        set the sub-dirs for training and validation data.
        """
        self.data_dir  = new_data_dir
        self.train_dir = os.path.join(new_data_dir, "train2017")
        self.val_dir   = os.path.join(new_data_dir, "val2017")
        return


########################################################################
    def generate_vocabulary(self):

        filename = self.data_dir + 'vocabulary/vocabulary.pickle'
        if not os.path.isfile(filename):
            generateVocabulary.generateVocabulary(self.data_dir, self.train_records_list, self.val_records_list)
        else:
            print('The file "vocabulary.pickle" has already been produced.')
        return

########################################################################
    def maybe_download_and_extract_vgg16weights(self):
        """
        Download and extract the VGG16 weights if the data-files don't
        already exist in data_dir.
        """
        # Base-URL for the data-sets on the internet.

        url = 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'

        print("Downloading " + url)
        downloadCoco.maybe_download_and_extract(url=url, download_dir=self.data_dir+'CNN/')


########################################################################
    def maybe_download_and_extract_coco(self):
        """
        Download and extract the COCO data-set if the data-files don't
        already exist in data_dir.
        """
        # Base-URL for the data-sets on the internet.
        data_url = "http://images.cocodataset.org/"

        # Filenames to download from the internet.
        filenames = ["zips/train2017.zip", "zips/val2017.zip",
                     "annotations/annotations_trainval2017.zip"]

        # Download these files.
        for filename in filenames:
            # Create the full URL for the given file.
            url = data_url + filename
            print("Downloading " + url)
            downloadCoco.maybe_download_and_extract(url=url, download_dir=self.data_dir)

########################################################################

    def load_records(self, trainSet=True):
        if trainSet:
            # Training-set.
            filename = "captions_train2017.json"
        else:
            # Validation-set.
            filename = "captions_val2017.json"

        # Full path for the data-file.
        path = os.path.join(self.data_dir, "annotations", filename)

        # Load the file.
        with open(path, "r", encoding="utf-8") as file:
            data_raw = json.load(file)

        # Convenience variables.
        images      = data_raw['images']
        annotations = data_raw['annotations']

        # Initialize the dict for holding our data.
        # The lookup-key is the image-id.
        records = dict()

        # Collect all the filenames for the images.
        for image in images:
            # Get the id and filename for this image.
            image_id = image['id']
            filename = image['file_name']

            # Initialize a new data-record.
            record = dict()

            # Set the image-filename in the data-record.
            record['filename'] = filename

            # Initialize an empty list of image-captions
            # which will be filled further below.
            record['captions'] = list()

            # Save the record using the the image-id as the lookup-key.
            records[image_id] = record

        # Collect all the captions for the images.
        for ann in annotations:
            # Get the id and caption for an image.
            image_id = ann['image_id']
            caption  = ann['caption']

            # Lookup the data-record for this image-id.
            # This data-record should already exist from the loop above.
            record = records[image_id]

            # Append the current caption to the list of captions in the
            # data-record that was initialized in the loop above.
            record['captions'].append(caption)

        # Convert the records-dict to a list of tuples.
        coco_records_list = [(key, record['filename'], record['captions'])
                        for key, record in sorted(records.items())]

        # Convert the list of tuples to separate tuples with the data.
        #ids, filenames, captions = zip(*records_list)

        if trainSet:
            self.trainNumbOfImages = len(coco_records_list)
            self.train_records_list = coco_records_list
        else:
            self.valNumbOfImages = len(coco_records_list)
            self.val_records_list = coco_records_list
        return

    def load_image(self, path):
        """
        Load the image from the given file-path and resize it
        to the given size if not None.
        """

        # Load the image using PIL.
        img = Image.open(path)

        # Resize image if desired.
        img = img.resize(size=self.imgSize, resample=Image.LANCZOS)

        # Convert image to numpy array.
        img = np.array(img)

        # Convert 2-dim gray-scale array to 3-dim RGB array.
        if (len(img.shape) == 2):
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        img = self.vgg_preprocessing(img)
        return img

    def saveFc7AsPickle(self, fc7_val, path_list, original_captions_list, vocabularyDict, saveDir):

        wordToToken = vocabularyDict['wordToToken']

        #Iterate through images
        for ii in range(len(path_list)):
            original_captions = original_captions_list[ii]
            captionsAsTokens = []
            captions         = []

            # Iterate through each caption
            for jj in range(len(original_captions)):
                original_caption = original_captions[jj]
                tokenList = []
                captionList = []
                tokenList.append(wordToToken['ssss'])
                captionList.append('ssss')

                # Convert to lowercase and spilt based on spaces
                original_caption = original_caption.lower()
                original_caption = original_caption.split(' ')
                for kk in range(len(original_caption)):
                    word = original_caption[kk]

                    # Remove all special characters and add to vocabulary
                    word = ''.join(e for e in word if e.isalnum())
                    if word != '':
                        tokenList.append(wordToToken[word])
                        captionList.append(word)

                        #Add "end token" at the end of the caption
                        if kk==len(original_caption)-1:
                            tokenList.append(wordToToken['eeee'])
                            captionList.append('eeee')

                captionsAsTokens.append(tokenList)
                captions.append(captionList)

            dataDict = {}
            dataDict['vgg16_fc7'] = fc7_val[ii, :]
            dataDict['imgPath']   = path_list[ii]
            dataDict['original_captions'] = original_captions_list[ii]
            dataDict['captionsAsTokens']  = captionsAsTokens
            dataDict['captions']          = captions

            str = saveDir+path_list[ii][:-4] + '.pickle'
            with open(str, 'wb') as handle:
                pickle.dump(dataDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def train_next_batch(self):
        iter         = self.trainImgIter
        batchSize    = self.trainBatchSize
        records_list = self.train_records_list
        img_list  = []
        path_list = []
        captions = []
        for ii in range(batchSize):
            if iter + ii < self.trainNumbOfImages:
                path = records_list[iter+ii][1]
                captions.append(records_list[iter+ii][2])
                path_list.append(path)
                img  = self.load_image(self.train_dir+'/'+path)


                img_list.append(img)
        self.trainImgIter = iter + batchSize
        return np.asarray(img_list), path_list, captions

    def val_next_batch(self):
        iter         = self.valImgIter
        batchSize    = self.valBatchSize
        records_list = self.val_records_list
        img_list  = []
        path_list = []
        captions = []
        for ii in range(batchSize):
            if iter + ii < self.valNumbOfImages:
                path = records_list[iter + ii][1]
                captions.append(records_list[iter + ii][2])
                path_list.append(path)
                img = self.load_image(self.val_dir + '/' + path)
                img_list.append(img)
        self.valImgIter = iter + batchSize
        return np.asarray(img_list), path_list, captions

    def vgg_preprocessing(self, img):
        rgb_means = np.array([[[123.68, 116.779, 103.939]]])
        img = img-rgb_means
        return img


    def produceVgg16Fc7(self):
        # Define placeholders for being able to feed data to the tensorflow graph
        data = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32, name='data')

        fc7 = CNN_network.vgg_16(inputs=data, is_training=False)

        # start session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        # Restore the pretrained weights
        trainable_variables = tf.trainable_variables()
        restorer = tf.train.Saver(trainable_variables)
        restorer.restore(sess, self.data_dir+'CNN/'+'vgg_16.ckpt')

        # restore vocabulary dict
        vocabularyDict = generateVocabulary.loadVocabulary(self.data_dir)

        saveDirTrain = self.data_dir + '/Train2017_vgg16_fc7/'
        saveDirVal  = self.data_dir + '/Val2017_vgg16_fc7/'

        if not os.path.exists(saveDirTrain):
            os.makedirs(saveDirTrain)
        if not os.path.exists(saveDirVal):
            os.makedirs(saveDirVal)

        if (os.listdir(saveDirTrain) == []) or (os.listdir(saveDirVal) == []):
            # Store all fc7 layers as pickel objects.
            for i in trange(int(np.ceil(self.trainNumbOfImages / self.trainBatchSize)),
                            desc='Generate: Train pickle files', leave=True):
                data_batch, path_list, captions = self.train_next_batch()
                fc7_val = sess.run(fc7, {data: data_batch})
                self.saveFc7AsPickle(fc7_val, path_list, captions, vocabularyDict, saveDirTrain)

            for i in trange(int(np.ceil(self.valNumbOfImages / self.valBatchSize)),
                            desc='Generate: Val pickle files', leave=True):
                data_batch, path_list, captions = self.val_next_batch()
                fc7_val = sess.run(fc7, {data: data_batch})
                self.saveFc7AsPickle(fc7_val, path_list, captions, vocabularyDict, saveDirVal)
        else:
            print("Pickle files have apparently already been produced.")
        return


###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################


class CocoVGG16FC7DataClass():
    def __init__(self, trainBatchSize, valBatchSize, truncated_backprop_length, vocabulary_size):
        self.data_dir = "data/coco/"

        self.trainBatchSize = trainBatchSize
        self.valBatchSize  = valBatchSize
        self.truncated_backprop_length = truncated_backprop_length

        self.vocabulary_size = vocabulary_size

        self.trainNumbOfImages = 0
        self.valNumbOfImages  = 0

        self.trainIter = 0
        self.valIter  = 0
        self.trainCaptionIter = []
        self.valCaptionIter = []

        self.trainPathList = []
        self.valPathList = []
        return

    def setDatapath(self, path):
        self.data_dir = path
        return

    def getfilePaths(self):
        self.trainPathList = glob.glob(self.data_dir + '/Train2017_vgg16_fc7/*')
        self.valPathList  = glob.glob(self.data_dir + '/Val2017_vgg16_fc7/*')

        self.trainNumbOfImages = len(self.trainPathList)
        self.valNumbOfImages  = len(self.valPathList)

        self.trainCaptionIter = np.zeros((self.trainNumbOfImages))
        self.valCaptionIter  = np.zeros((self.valNumbOfImages))

        self.trainIterPerEpoch = int(np.ceil(self.trainNumbOfImages/self.trainBatchSize))
        self.valIterPerEpoch  = int(np.ceil(self.valNumbOfImages / self.valBatchSize))
        return

    def getCaptionMatix(self, captionsAsTokens):
        # find the length sequence and create correspinding captionMatix

        batchSize  = len(captionsAsTokens)
        seqLengths = [len(tokens) for tokens in captionsAsTokens]
        maxSeqLen  = max(seqLengths)

        divisionCount = int(np.ceil((maxSeqLen-1)/self.truncated_backprop_length))
        maxLength     = self.truncated_backprop_length*divisionCount + 1

        captionMatix  = np.zeros((batchSize, maxLength), dtype=int)
        weightMatrix  = np.zeros((batchSize, maxLength), dtype=int)

        mask               = np.arange(maxLength) < np.array(seqLengths)[:, None]
        captionMatix[mask] = np.concatenate(captionsAsTokens)
        weightMatrix[mask] = 1

        #set all words with index larger then "vocabulary_size" to "UNK" unknown word -> index=2
        captionMatix[captionMatix >= self.vocabulary_size] = 2

        yTokens  = captionMatix[:,1:].reshape((batchSize, self.truncated_backprop_length, divisionCount), order='F')
        yWeights = weightMatrix[:,1:].reshape((batchSize, self.truncated_backprop_length, divisionCount), order='F')

        xTokens  = captionMatix[:,:-1].reshape((batchSize, self.truncated_backprop_length, divisionCount), order='F')
        xWeights = weightMatrix[:,:-1].reshape((batchSize, self.truncated_backprop_length, divisionCount), order='F')

        # id = 0
        # print(captionsAsTokens[id])
        # print('\n')
        # print(xTokens[id ,: ,0])
        # print(xTokens[id, :, 1])
        # print('\n')
        # print(xWeight[id ,: ,0])
        # print(xWeight[id, :, 1])
        # print('\n')
        # print(yTokens[id, :, 0])
        # print(yTokens[id, :, 1])
        # print(yWeight[id ,: ,0])
        # print(yWeight[id, :, 1])
        # print('\n')
        return xTokens, yTokens, yWeights

    def train_next_batch(self, getImages=False):
        batchSize = self.trainBatchSize
        pathList  = self.trainPathList
        iter      = self.trainIter
        captionIter = self.trainCaptionIter
        vgg_fc7           = []
        orig_captions     = []
        captions          = []
        captionsAsTokens  = []
        imgPaths          = []
        for ii in range(batchSize):
            if iter + ii < self.trainNumbOfImages:
                id = iter+ii
                #print(id)
                captionId = captionIter[id]
                with open(pathList[id], "rb") as input_file:
                    data = pickle.load(input_file)
                tmpOrigCaption      = data['original_captions']
                tmpCaption          = data['captions']
                tmpCaptionsAsTokens = data['captionsAsTokens']
                imgPaths.append(data['imgPath'])
                vgg_fc7.append(data['vgg16_fc7'])

                if len(tmpOrigCaption) <= captionId:
                    orig_captions.append(tmpOrigCaption[captionId])
                    captions.append(tmpCaption[captionId])
                    captionsAsTokens.append(tmpCaptionsAsTokens[captionId])
                    captionId = captionId + 1
                else:
                    orig_captions.append(tmpOrigCaption[0])
                    captions.append(tmpCaption[0])
                    captionsAsTokens.append(tmpCaptionsAsTokens[0])
                    captionId = 1
                self.trainCaptionIter[id] = captionId
            else:
                self.trainIter = -batchSize
        self.trainIter = self.trainIter + batchSize

        xTokens, yTokens, yWeights = self.getCaptionMatix(captionsAsTokens)
        # return imgPaths
        return np.array(vgg_fc7), xTokens, yTokens, yWeights, captions, orig_captions, imgPaths

    def val_next_batch(self, getImages=False):
        batchSize = self.valBatchSize
        pathList  = self.valPathList
        iter      = self.valIter
        captionIter = self.valCaptionIter
        vgg_fc7           = []
        orig_captions     = []
        captions          = []
        captionsAsTokens  = []
        imgPaths          = []
        for ii in range(batchSize):
            if iter + ii < self.valNumbOfImages:
                id = iter+ii
                #print(id)
                captionId = captionIter[id]
                with open(pathList[id], "rb") as input_file:
                    data = pickle.load(input_file)
                tmpOrigCaption      = data['original_captions']
                tmpCaption          = data['captions']
                tmpCaptionsAsTokens = data['captionsAsTokens']
                imgPaths.append(data['imgPath'])
                vgg_fc7.append(data['vgg16_fc7'])

                orig_captions.append(tmpOrigCaption)

                if len(tmpOrigCaption) <= captionId:
                    captions.append(tmpCaption[captionId])
                    captionsAsTokens.append(tmpCaptionsAsTokens[captionId])
                    captionId = captionId + 1
                else:
                    captions.append(tmpCaption[0])
                    captionsAsTokens.append(tmpCaptionsAsTokens[0])
                    captionId = 1
                self.valCaptionIter[id] = captionId
            else:
                self.valIter = -batchSize
        self.valIter = self.valIter + batchSize

        xTokens, yTokens, yWeights = self.getCaptionMatix(captionsAsTokens)
        return np.array(vgg_fc7), xTokens, yTokens, yWeights, captions, orig_captions, imgPaths

    def getRandomValBatch(self, batchSize):
        pathList  = self.valPathList
        vgg_fc7           = []
        orig_captions     = []
        captions          = []
        captionsAsTokens  = []
        imgPaths          = []
        for ii in range(batchSize):
            id = np.random.randint(low=0, high=len(pathList)-1)
            captionId = 0
            with open(pathList[id], "rb") as input_file:
                data = pickle.load(input_file)
            tmpOrigCaption      = data['original_captions']
            tmpCaption          = data['captions']
            tmpCaptionsAsTokens = data['captionsAsTokens']
            imgPaths.append(data['imgPath'])
            vgg_fc7.append(data['vgg16_fc7'])
            orig_captions.append(tmpOrigCaption)
            captions.append(tmpCaption[captionId])
            captionsAsTokens.append(tmpCaptionsAsTokens[captionId])
        xTokens, yTokens, yWeights = self.getCaptionMatix(captionsAsTokens)
        return np.array(vgg_fc7), xTokens, yTokens, yWeights, captions, orig_captions, imgPaths



