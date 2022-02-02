
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUiType

import ip_model as ip
import numpy as np
import cv2 as cv
import os
import sys

MainUI,_ = loadUiType('main.ui')

class Main(QMainWindow, MainUI):
    
    directory = str
    image = None

    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.directory = 'null'
        self.handle_UI()
        self.handle_buttons()


    def handle_UI(self):
        self.setWindowTitle('Image Processing Project')
        self.setFixedSize(1255,858)
        
    
    def handle_buttons(self):
        
        ''' File Handling Buttons '''
        self.open_file.clicked.connect(self.handle_open)
        self.save_file.clicked.connect(self.handle_save_result)
        self.exit.clicked.connect(QApplication.instance().quit)
        
        ''' Brightness & Contrast Adjustment Buttons '''
        self.color.clicked.connect(self.colored)
        self.gray.clicked.connect(self.gray_color)
        
        
        ''' Adding Noise Buttons '''
        self.salt_pepper.clicked.connect(self.salt_pepper_noise)
        self.gaussian_n.clicked.connect(self.gaussian_noise)
        self.poisson_n.clicked.connect(self.poisson_noise)
        
        ''' Point Transformation Buttons '''
        self.brightness.clicked.connect(self.brightness_adjustment)
        self.contrast.clicked.connect(self.contrast_adjustment)
        self.hist.clicked.connect(self.histogram)
        self.hist_equalization.clicked.connect(self.histogram_equalization)
        
        ''' Local Transformation Buttons '''
        self.lpf.clicked.connect(self.low_pass_filter)
        self.hpf.clicked.connect(self.high_pass_filter)
        self.median.clicked.connect(self.median_filter)
        self.average.clicked.connect(self.average_filter)
        self.laplacian_f.clicked.connect(self.laplacian_filter)
        self.gaussian_f.clicked.connect(self.gaussian_filter)
        self.v_sobel.clicked.connect(self.vertical_sobel)
        self.h_sobel.clicked.connect(self.horizontal_sobel)
        self.v_prewitt.clicked.connect(self.vertical_prewitt)
        self.h_prewitt.clicked.connect(self.horizontal_prewitt)
        self.log.clicked.connect(self.laplacian_of_gaussian)
        self.canny.clicked.connect(self.canny_method)
        self.zero_crossing.clicked.connect(self.zero_cross)
        self.skeleton.clicked.connect(self.skeleton_filter)

        ''' Global Transformation Buttons '''
        self.line_hough.clicked.connect(self.hough_line_transform)
        self.circle_hough.clicked.connect(self.hough_circle_transform)
        self.dilate.clicked.connect(self.dilation)
        self.erode.clicked.connect(self.erosion)
        self.open.clicked.connect(self.opening)
        self.close.clicked.connect(self.closing)


    def handle_save_result(self):
        if self.image == None:
            message = QMessageBox.warning(self, 'Warning', "You must print a result image first!!", QMessageBox.Ok, QMessageBox.Ok)

        else:
            save_location = QFileDialog.getSaveFileName(self, caption='Save As', directory= os.path.dirname(__file__)+'Result Image', filter="JPEG(*.jpg *.jpeg);;PNG(*.png);;All Files(*.*)")[0]
            self.image.save(save_location)


    def handle_open(self):
        self.intialize_radio_buttons()
        self.image = None
        self.photo_3.clear()
        self.directory = QFileDialog.getOpenFileName(self, caption='Open File', directory= os.path.dirname(__file__), filter="All Files(*.*);;JPEG(*.jpg *.jpeg);;PNG(*.png)")[0]
        self.photo_1.setPixmap(QPixmap(self.directory))        

    
    def intialize_radio_buttons(self):
        self.intialize_convert_RB()
        self.intialize_add_noise_RB()
        self.intialize_edge_detection_RB()

    
    def pop_up_message(self):
        message = QMessageBox.warning(self, 'Warning', "You must open an image first!!", QMessageBox.Ok, QMessageBox.Ok)
        self.intialize_radio_buttons()
        return False

        
    def check_directory(self):
        check_result = True
        if self.directory == 'null' or '' or None :
            check_result = self.pop_up_message()
            
        return check_result


    def show_image(self, img):
        
        if img.ndim == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            self.image = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped()

        else:
            self.image = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped()
        
        self.photo_3.setPixmap(QPixmap.fromImage(self.image))
        

    def colored(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_add_noise_RB()
            self.intialize_edge_detection_RB()
            
            img = ip.load_img(self.directory)
            self.show_image(img)

    def gray_color(self):
        check_result = self.check_directory()
        
        if check_result == True:        
            self.intialize_add_noise_RB()
            self.intialize_edge_detection_RB()

            img = ip.convert_gray(self.directory)
            self.show_image(img)


    ''' Add Noise '''
    def salt_pepper_noise(self):
        check_result = self.check_directory()
                
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_edge_detection_RB()

            img = ip.salt_pepper_noise(self.directory)
            self.show_image(img)

    def gaussian_noise(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_edge_detection_RB()

            img = ip.gaussian_noise(self.directory)
            self.show_image(img)

    def poisson_noise(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_edge_detection_RB()

            img = ip.poisson_noise(self.directory)
            self.show_image(img)


    ''' Local Transformation '''
    def brightness_adjustment(self):
        check_result = self.check_directory()
    
        if check_result == True:
            img = ip.brightness_adjustment(self.directory)
            self.show_image(img)


    def contrast_adjustment(self):
        check_result = self.check_directory()

        if check_result == True:
            img = ip.contrast_adjustment(self.directory)
            self.show_image(img)

    def histogram(self):
        check_result = self.check_directory()
        
        if check_result == True:
            ip.histogram(self.directory)
        
    def histogram_equalization(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.histogram_equalization(self.directory)
            self.show_image(img)

    def low_pass_filter(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.low_pass_filter(self.directory)
            self.show_image(img)

    def high_pass_filter(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.high_pass_filter(self.directory)
            self.show_image(img)

    def median_filter(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.median(self.directory)
            self.show_image(img)

    def average_filter(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.average(self.directory)
            self.show_image(img)

    def laplacian_filter(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_add_noise_RB()
            
            img = ip.laplacian(self.directory)
            self.show_image(img)

    def gaussian_filter(self):
        check_result = self.check_directory()

        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_add_noise_RB()

            img = ip.gaussian(self.directory)
            self.show_image(img)

    def vertical_sobel(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_add_noise_RB()
            
            img = ip.vertical_sobel(self.directory)
            self.show_image(img)

    def horizontal_sobel(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_add_noise_RB()
            
            img = ip.horizontal_sobel(self.directory)
            self.show_image(img)


    def vertical_prewitt(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_add_noise_RB()
            
            img = ip.vertical_prewitt(self.directory)
            self.show_image(img)

    def horizontal_prewitt(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_add_noise_RB()
            
            img = ip.horizontal_prewitt(self.directory)
            self.show_image(img)


    def laplacian_of_gaussian(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_add_noise_RB()
            
            img = ip.laplacian_of_gaussian(self.directory)
            self.show_image(img)

    def canny_method(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_add_noise_RB()
            
            img = ip.canny(self.directory)
            self.show_image(img)

    def zero_cross(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_add_noise_RB()
            
            img = ip.zero_cross(self.directory)
            self.show_image(img)

    def skeleton_filter(self):
        check_result = self.check_directory()
        
        if check_result == True:
            self.intialize_convert_RB()
            self.intialize_add_noise_RB()
            
            img = ip.skeleton(self.directory)
            self.show_image(img)

    def hough_line_transform(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.Hough_Line_Transform(self.directory)
            self.show_image(img)

    def hough_circle_transform(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.Hough_Circle_Transform(self.directory)
            self.show_image(img)

    def dilation(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.dilation(self.directory)
            self.show_image(img)

    def erosion(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.erosion(self.directory)
            self.show_image(img)

    def opening(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.opening(self.directory)
            self.show_image(img)

    def closing(self):
        check_result = self.check_directory()
        
        if check_result == True:
            img = ip.closing(self.directory)
            self.show_image(img)


    def intialize_convert_RB(self):
        self.color.setAutoExclusive(False)
        self.color.setChecked(False)
        self.color.setAutoExclusive(True)

        self.gray.setAutoExclusive(False)
        self.gray.setChecked(False)
        self.gray.setAutoExclusive(True)

    def intialize_add_noise_RB(self):
        self.salt_pepper.setAutoExclusive(False)
        self.salt_pepper.setChecked(False)
        self.salt_pepper.setAutoExclusive(True)

        self.gaussian_n.setAutoExclusive(False)
        self.gaussian_n.setChecked(False)
        self.gaussian_n.setAutoExclusive(True)

        self.poisson_n.setAutoExclusive(False)
        self.poisson_n.setChecked(False)
        self.poisson_n.setAutoExclusive(True)

    def intialize_edge_detection_RB(self):
        self.laplacian_f.setAutoExclusive(False)
        self.laplacian_f.setChecked(False)
        self.laplacian_f.setAutoExclusive(True)

        self.gaussian_f.setAutoExclusive(False)
        self.gaussian_f.setChecked(False)
        self.gaussian_f.setAutoExclusive(True)
        
        self.v_prewitt.setAutoExclusive(False)
        self.v_prewitt.setChecked(False)
        self.v_prewitt.setAutoExclusive(True)

        self.h_prewitt.setAutoExclusive(False)
        self.h_prewitt.setChecked(False)
        self.h_prewitt.setAutoExclusive(True)

        self.v_sobel.setAutoExclusive(False)
        self.v_sobel.setChecked(False)
        self.v_sobel.setAutoExclusive(True)

        self.h_sobel.setAutoExclusive(False)
        self.h_sobel.setChecked(False)
        self.h_sobel.setAutoExclusive(True)

        self.log.setAutoExclusive(False)
        self.log.setChecked(False)
        self.log.setAutoExclusive(True)

        self.canny.setAutoExclusive(False)
        self.canny.setChecked(False)
        self.canny.setAutoExclusive(True)

        self.skeleton.setAutoExclusive(False)
        self.skeleton.setChecked(False)
        self.skeleton.setAutoExclusive(True)

        self.zero_crossing.setAutoExclusive(False)
        self.zero_crossing.setChecked(False)
        self.zero_crossing.setAutoExclusive(True)


def main():
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())



if __name__ == '__main__':
    main()