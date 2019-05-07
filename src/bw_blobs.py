import numpy as np
import scipy.ndimage.filters as filter
from scipy import ndimage
import cv2
import os
import shutil

class BW_Blobs:
    def __init__(self):
        self.debug = True
        self.rmax_sz = 5

    def show_image(self, in_image, name='image'):
        cv2.imshow(name, in_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def imreconstruct(self, marker, mask, conn=8):
        if conn==8:
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        else:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        recon1 = marker
        recon1_old = np.zeros(np.shape(recon1), np.uint8)
        count = 0
        while sum(sum(recon1 - recon1_old)) != 0:
            recon1_old = recon1
            recon1 = cv2.dilate(recon1, se)
            bw = recon1 > mask
            recon1[bw] = mask[bw]
            count +=1
            print('difference = ', sum(sum(recon1 - recon1_old)))
        print('total loop count = ', count)
        return recon1

    def imhmax(self, img, h, conn=8):
        mask = img
        marker = np.zeros(np.shape(img), np.uint8)
        [row,col] = mask.shape[:2]
        for r in range(row):
            for c in range(col):
                clipped = int(mask[r][c] - h)
                if clipped < 0:
                    clipped = 0
                marker[r][c] = clipped

        # marker = mask - h
        bw = self.imreconstruct(marker, mask, conn)
        return bw

    def imregionalmax(self, img):
        [r,c] = img.shape[:2]
        size = [(int)(r/self.rmax_sz), (int)(c/self.rmax_sz)]
        localmax = filter.maximum_filter(img, size, mode='constant')
        mask = (img == localmax)
        maximg = np.zeros(np.shape(img), np.uint8)
        maximg[mask] = 255
        return maximg

    def imextendedmax(self, img, h):
        bw_temp = self.imhmax(img, h)
        self.show_image(bw_temp,'imhmax')
        bw = self.imregionalmax(bw_temp)
        return bw

    def imclose(self, blobs):
        kernel = np.ones((3,3), np.uint8)
        bw_close = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, kernel=kernel)
        return bw_close

    def imfill(self, blobs):
        im_floodfill = blobs.copy()
        r,c = blobs.shape[:2]
        mask = np.zeros((r+2,c+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_out = blobs | im_floodfill
        return im_out

    def bwareaopen(self, blobs, a):
        kernel = np.ones((3, 3), np.uint8)
        bwopen = ndimage.binary_opening(blobs, structure=kernel)
        openimg = np.zeros(np.shape(blobs), np.uint8)
        openimg[bwopen] = 255
        _, contours, _ = cv2.findContours(openimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            print(area)
            if area > a:
                continue
            else:
                for indx in cnt:
                    openimg[indx[0]][indx[1]] = 0
        return openimg


    def gray2bw(self, image, graythresh, areathresh):
        bwblobs1 = self.imextendedmax(image, graythresh)
        self.show_image(bwblobs1,'extendedmax')
        bwblobs2 = self.imclose(bwblobs1)
        self.show_image(bwblobs2, 'imclose')
        bwblobs3 = self.imfill(bwblobs2)
        self.show_image(bwblobs3, 'imfill')
        bwblobs4 = self.bwareaopen(bwblobs3, areathresh)
        self.show_image(bwblobs4, 'bwareaopen')


if __name__ == "__main__":
    input_image = './../sample_data/glass.png'
    bw_blobs = BW_Blobs()
    image = cv2.imread(input_image, 0)
    bw_blobs.show_image(image, 'input')
    blobs = bw_blobs.gray2bw(image, 80, 30)

    print('Done')

