import cv2
import numpy as np
import os

class DATA(object):
    def data_sort(self):
        data = []
        for img in os.listdir("results"):
            path = os.path.join("results", img)
            img_data = cv2.imread(path)
            img_data = cv2.resize(img_data, (299, 299))
            data.append([np.array(img_data), img.split('.')[-3]])
        return data,img_data
    
    def video_frames(self, data):
        test = data[:1]
        X_test = np.array([i[0] for i in test]).reshape(-1, 299, 299, 3)
        return X_test