from __future__ import division

import collections
from skimage.transform import resize

import face_alignment
from facenet_pytorch import fixed_image_standardization
from torchvision import transforms
from eyecolor import eye_color
from utils import *
import cv2 

ageProto="./models/age_deploy.prototxt"
ageModel="./models/age_net.caffemodel"
genderProto="./models/gender_deploy.prototxt"
genderModel="./models/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# ageList=['(0-2)', '(4-6)',  '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
# ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)





class Face(object):
    def __init__(self, faceImage, fa = None, device = None, label = 'NONE'):
        self.pilImage = faceImage
        self.label = label
        self.imgCv2 =  np.asarray(self.pilImage)[:, :, ::-1].copy()
        cv2.imwrite("t.jpg", self.imgCv2)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.device_str = 'cpu' if self.device == torch.device('cpu') else 'cuda:0'
        self.fa = fa if fa is not None else face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device = self.device_str)
        lms = self.fa.get_landmarks(self.imgCv2[:, :, ::-1])[-1]
        self.lms2d = np.array([[p[0], p[1]] for p in lms])
        self.lms2dnormed = (lms - np.min(lms)) / np.max(lms) * 256
        # print(lms)
        self.transforms_ = transforms.Compose([
            transforms.Resize((256, 256)),
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        self.faceTensor = self.transforms_(faceImage).to(self.device)
        self.eyecolor = eye_color(self.imgCv2.copy(), lms)
        blob=cv2.dnn.blobFromImage(self.imgCv2, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        self.gender = gender

        
    
    def show_face(self):
        return self.pilImage
        # plt.show()
    def show_lms(self):
        im = vis_landmark_on_img(self.lms2d, img=self.imgCv2)
        return Image.fromarray(im[:, :, ::-1].astype(np.uint8))
    def get_params(self):
        return {
            "landmarks" : ' '.join([str(i) for i in self.lms2d.reshape(68*2).tolist()]),
            "landmarks_normed" : ' '.join([str(i) for i in self.lms2dnormed.reshape(68*2).tolist()]),
            "eye_color" : self.eyecolor, 
            "age" : 0,
            "gender" : self.gender
        }
