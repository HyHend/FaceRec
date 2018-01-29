import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

""" Note: Made extensive use of https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py
"""

class FaceRecogniser:
    """ Example usage:
    
    Train:
    fr = FaceRecogniser(predictorPath, facerecModelPath)
    fr.train(trainImagesPath, labelsFilePath)
    
    Test:
    result = fr.recognise(testImagesPath)
    
    Downloads:
    predictorPath: http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
    facerecModelPath: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    """
    def __init__(self, predictorPath, facerecModelPath):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(predictorPath)
        self.facerec = dlib.face_recognition_model_v1(facerecModelPath)
        
    def train(self, trainImagesPath, labelsFilePath):
        """Uses given images and labels to create labeled groups of faces
        These groups can be used to recognise new faces with
        """
        labels = self.read_labels(labelsFilePath)
        images = self.process_images_in_folder(trainImagesPath, labels)
        self.faces = self.get_all_faces_from_images(images)
        self.groups = self.process_faces(self.faces, 
                                         self.faces, 
                                         groups=[], 
                                         threshold=0.6, 
                                         maxCompareInGroup=4)
        print("Num groups: {0}".format(len(self.groups)))
    
    def recognise(self, recogniseImagesPath):
        """ For each image in given folder
        retrieve faces
        compare faces with trained model
        update trained model with faces
        Returns faces with group labels when found
        """
        images = self.process_images_in_folder(recogniseImagesPath, {})
        recogniseFaces = self.get_all_faces_from_images(images)
        self.faces = {**self.faces, **recogniseFaces}
        
        self.groups = self.process_faces(recogniseFaces, 
                                   self.faces, 
                                   self.groups, 
                                   threshold=0.6, 
                                   maxCompareInGroup=4)
        print("Num groups: {0}".format(len(self.groups)))
        
        for key in recogniseFaces:
            face = recogniseFaces[key]
            result[key] = {'face':recogniseFaces[key], 
                           'annotation':self.find_face_in_groups(key)[0]}
    
    def read_labels(self, labelsFile):
        """ Reads labels from given file
        returns dict with key = image, value = label
        Expected file format: image_name,label\n
        
        Example:
        filename_1,Harry What
        filename_2,Sue Me
        """
        f = open(labelsFile,"r")
        labels = {}

        for line in f:
            line = line.strip().replace("\r","").replace("\n","").replace("jpg","").replace("JPG","")
            split = line.split(",")
            labels[split[0]] = split[1]
        return labels
    
    def process_image(self, img, label):
        """ Given image
        Returns an list of all faces, their location and their 128D vector
        If two faces have, between them, an Euclidian distance of less 
        than 0.6 they are likely to be the same person.

        Returns list of faces with:
            - the shape defining the face
            - the bounding box in the image
            - the face descriptor vector
            - the sub image showing the face
        """

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = self.detector(img, 1)
        faces = {}
        print("Number of faces detected: {}".format(len(dets)))

        #TODO only label face with largest box in image.
        # For now, don't label with more than 1 face in image
        if(len(dets) > 1) and label is not False:
            print("Multiple faces found in image. Ignoring label.")
            label = False

        # Process each face we found.
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = self.sp(img, d)

            # Compute the 128D vector that describes the face in img identified by
            # shape.
            face_descriptor = self.facerec.compute_face_descriptor(img, shape)
            # face_descriptor = facerec.compute_face_descriptor(img, shape, 100) # Higher accuracy (99.13 to 99.38, 100 times slower)

            # Cut out face from img. Store
            faceImg = img[d.top():d.bottom(), d.left():d.right()]

            faces[hash(faceImg.tostring())] = {
                'shape' : shape,
                'box': d,
                'vector': face_descriptor,
                'image': faceImg,
                'label': label
            }
        return faces

    def process_images_in_folder(self, folder, labels):
        """ Processes all jpg files in given folder.
        Returns list of images with found faces and their specifics
        """
        images = []
        for f in glob.glob(os.path.join(folder, "*.jpg")):
            print("Processing file: {}".format(f))
            img = io.imread(f)

            # Try to get label for image
            label = False
            name = f.replace(".jpg","").replace(".JPG","").split("/")[-1]
            try:
                label = labels[name]
            except:
                pass

            images.append({
                'filename': f,
                'image': img,
                'result': self.process_image(img, label)
            })
        return images

    def get_all_faces_from_images(self, images):
        """ Returns dict with faces and their vector
        """ 
        faces = {}
        for image in images:
            for face in image['result']:
                faces[face] = image['result'][face]
        return faces

    def compare_faces(self, a, b):
        """ Compares face a with face b
        Returns euclidian distance between a, b
        """
        x = np.array(list(a['vector']))
        y = np.array(list(b['vector']))
        return np.linalg.norm(x-y)

    def process_face(self, newFace, faceKey, allFaces, groups, threshold=0.6, maxCompareInGroup=4):
        """ Processes face corresponding to given faceKey
        Returns group it belongs to and the entire list of groups
        """
        # Random arrangement of groups
        # Uniformly distributes time probability of finding the face 
        x = [i for i in range(0,len(groups))]
        random.shuffle(x)
        lowestDistance = 1
        lowestDistanceGroup = 0
        for j in x:
            # Sample maxCompareInGroup random faces from group, compare
            compareIndices = random.sample(range(0, len(groups[j]['faces'])), min(maxCompareInGroup, len(groups[j]['faces'])))
            compoundDistance = 0
            for k in compareIndices:
                # Compare face with face in group. Update lowestDistance if closer
                distance = self.compare_faces(allFaces[groups[j]['faces'][k]], newFace)
                if distance < lowestDistance:
                    lowestDistance = distance
                    lowestDistanceGroup = j
        if lowestDistance < threshold:
            # Add face to group (Group corresponding to lowest distance is closer than threshold)
            groups[lowestDistanceGroup]['faces'].append(faceKey)

            # Set group label if existent in this image
            label = groups[lowestDistanceGroup]['label']
            if newFace['label'] is not False:
                label = newFace['label']

            groups[lowestDistanceGroup] = {
                        'label': label,
                        'faces': groups[lowestDistanceGroup]['faces']
                    }
            print("Added face to existing group: {0}".format(lowestDistanceGroup))
            return (groups[lowestDistanceGroup], groups)
        else: 
            groups.append({'label':newFace['label'], 'faces':[faceKey]})
            print("Adding new group..")
        return ({'label':newFace['label'], 'faces':[faceKey]}, groups)

    def process_faces(self, newFaces, allFaces, groups=[], threshold=0.6, maxCompareInGroup=4):
        """ Processes all faces in faces
        Returns list of groups
        """
        for faceKey in newFaces:
            res, groups = self.process_face(newFaces[faceKey], faceKey, allFaces, groups, threshold, maxCompareInGroup)
        return groups

    def find_face_in_groups(self, faceHash, groups):
        """ Given hash belonging to Face, 
        looks through groups to find to what group it belongs 
        Returns group label and group
        """
        for group in self.groups:
            if faceHash in group['faces']:
                return (group['label'], group)
        else:
            return (False, False)