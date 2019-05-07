import numpy as np
from src.bw_blobs import BW_Blobs
import cv2
import os
import imutils


class ShapeDetector:
    def __init__(self):
        self.debug = True
        self.detect_circles = True
        self.detect_rectangles = False
        self.rmax_sz = 5
        self.bw = BW_Blobs()
        self.result_dir = './../results/'

    def get_circles(self, img):
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #self.bw.show_image(th, 'adaptive')
        ed = cv2.Canny(th, 150, 200)
        #self.bw.show_image(ed, 'canny')
        kernel = np.ones((3,3), np.uint8)
        op = cv2.morphologyEx(ed, cv2.MORPH_OPEN, kernel=kernel)
        #self.bw.show_image(op, 'opening')
        #img = cv2.medianBlur(img, 5)
        circles = None
        param1, param2 = 100, 80
        while circles is None:
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=param1, param2=param2, minRadius=10, maxRadius=120) # 40
            if circles is not None:
                break
            param1 = param1 - 10
            param2 = param2 - 10

        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(cimg, (i[0],i[1]), i[2], (0,255,0),2)
            #cv2.circle(cimg, (i[0],i[1]), 2, (0,0,255),3)
        return cimg

    def get_probably_lines(self, img):
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #img = cv2.medianBlur(img, 5)
        ed = cv2.Canny(img, 50, 150, apertureSize=3)
        #self.bw.show_image(ed)
        min_length = 1
        max_gap = 500
        lines = cv2.HoughLinesP(ed, 1, np.pi/180,200,minLineLength=min_length,maxLineGap=max_gap)
        #print(lines)
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(cimg,(x1,y1),(x2,y2),(0,255,0),2)

        lines1 = cv2.HoughLines(img, 1, np.pi/180,100)
        rho, theta = [], []
        for line in lines1:
            r = line[0][0]
            t = line[0][1]
            rho.append(r)
            theta.append(t)
            a = np.cos(t)
            b = np.sin(t)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1*(-b))
            y1 = int(y0 + 1*(a))
            x2 = int(x0 - 1*(-b))
            y2 = int(y0 - 1*(a))
            #cv2.line(cimg, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return cimg


    def save_images(self, name, marked_image, shape):
        words = name.split("/")
        patch_words = words[-1].split(".")
        new_name = os.path.join(self.result_dir, (str(patch_words[0]) + '_localised_' + shape + '_.png'))
        cv2.imwrite(new_name, marked_image)

    def get_image_samples_from_dir(self, srcdir):
        for filename in os.listdir(srcdir):
            sample_name = os.path.join(srcdir, filename)
            print('Processing: ', sample_name)
            image = cv2.imread(sample_name, 0)
            res = imutils.resize(image, width=600)
            if self.detect_circles:
                cimg = self.get_circles(res)
                type = 'circles'
                self.save_images(sample_name, cimg, type)
            if self.detect_rectangles:
                cimg = self.get_probably_lines(res)
                type = 'lines'
                self.save_images(sample_name, cimg, type)




if __name__ == "__main__":
    input_image = './../sample_data/sample13.jpg'
    shape = ShapeDetector()
    image = cv2.imread(input_image, 0)
    res = imutils.resize(image, width=600)
    #shape.bw.show_image(image, 'input')
    #cimg = shape.get_circles(res)
    #shape.bw.show_image(cimg, 'cimg')
    #shape.save_images(input_image, cimg)
    srcdir = './../sample_data/'
    shape.get_image_samples_from_dir(srcdir)
    print('Done')

