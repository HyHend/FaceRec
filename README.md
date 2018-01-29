# FaceRec
Face recognition with python3, dlib.
Allows to be trained on a set of images and an csv with labels.
Training will find faces in an image and group them together if it's the same person.
Pictures with only one face in them and a label line in the csv will allow the person group to receive given label.

The recognising step will recognise faces in the given images. Based on the trained groups, labels, it will label the new faces accordingly.

The DLIB image recognition has an accuracy of 99.13% (99.36%, along with an performance hit. Given some parameter tuning.)

### Example usage:

Train:
`fr = FaceRecogniser(predictorPath, facerecModelPath)`
`fr.train(trainImagesPath, labelsFilePath)`

Test:
`result = fr.recognise(testImagesPath)`

Downloads:
- predictorPath: `http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2`
- facerecModelPath: `http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2`

### Note
Made extensive use of https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py