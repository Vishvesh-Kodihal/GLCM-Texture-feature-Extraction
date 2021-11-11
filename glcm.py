
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import normalize
import time
from io import BytesIO
import numba
import numpy as np
from sklearn.cluster import KMeans
from tkinter import *
import tkinter as tk
# import tkinter 

a = np.zeros((256,256))

for i in range(256):
    for j in range(256):
        b = (i-j)**2
        a[i][j]=b

b = np.zeros((256,256))
for i in range(256):
    for j in range(256):
        b[i][j] = 1/(1+ (i - j)**2)
 
    
"""270 Degree GLCM"""


class GLCM_FeatureExtraction_270:
    
    def __init__(self,image,filter_size=None):
        self.img_array=image
        self.filter_size=filter_size
        self.clustered_image=None
    
    def imagePadding(self,band,paddingValue):
        ##vertical Top padding
        ##vertical bottom padding
        band =  np.insert(band,len(band),band[-1]*np.ones(paddingValue)[:,None],axis = 0)
        ##Horizontal left padding
        band = np.insert(band, 0, band[:,0]*np.ones(paddingValue)[:,None], axis=1)
        ##Horizontal right padding
        band = np.insert(band, len(band[0]), band[:,-1]*np.ones(paddingValue)[:,None], axis=1)
        return band
    
    def angular_second_moment(self,numarray):
#         print(type(numarray))
        arr = numarray**2.0
    #     sum of numpy array
        arsum = np.sum(arr)
        # print(a)
        return arsum
    
    # assuming numpy array

    def entropy_np(self,num_array):
        numlog = np.log(1/(num_array+0.0001))
        entrnp = num_array*numlog
        entrnp_sum = np.sum(entrnp)
        return entrnp_sum
    # contrast

    def con_arr(self,input_matrix):
        con_sum = np.sum(a*input_matrix)
        return con_sum
        
    def idm(self,input_matrix):
        idm_sum = np.sum(b*input_matrix)
        return idm_sum


    
    def transpose_addition(self, temp):
        temp_arry0=(np.array(temp)).transpose()
        temp_arry0 = temp + temp_arry0
        # sum = np.sum(temp0)
        temp_arry0 = sklearn.preprocessing.normalize(temp_arry0,norm='l2')  
#         print(type(temp_arry0))
#         print(temp_arry0)
        return temp_arry0
    # @numba.jit(nopython=True)
    def calculate_co_occurence_matrix(self,neighbourhoodMatrix):
        
        #create all the required temp arrays:
        
        rows=len(neighbourhoodMatrix)
        cols=len(neighbourhoodMatrix[0])
        #print(rows)
        #print(cols)
        #neighbourhoodMatrix=[[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]]
        temp90   = np.zeros((256, 256))
        
        
        #calculate the 90 degree co-occurence matrix:
        
        for i in range(rows):
            for j in range(cols):
                if  i+1<rows :
                    ind1=neighbourhoodMatrix[i][j]
                    ind2=neighbourhoodMatrix[i+1][j]
                    temp90[ind1][ind2]=temp90[ind1][ind2]+1
        # glcm90 = self.transpose_addition(temp90)
        glcm270 = temp90
        return glcm270
    


    
    def convolution(self,band):
        if (self.filter_size % 2 == 0):
            raise Exception("K should be odd number.")
            
        # paddingValue = int((self.filter_size - 1)/2)
        paddingValue = 0
        imageBand = self.imagePadding(band,paddingValue)
        arrayshape = imageBand.shape
        print(arrayshape)
        outputband90 = []
        output = []
        output_ans1 = []
        output_ans2 = []
        output_idm = []
        output_final_270 = []
        for rowNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
            outputbandCol90 = []

            for colNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
                neighbourhoodMatrix = imageBand[rowNo:rowNo + self.filter_size,colNo:colNo + self.filter_size]
                ## writing code for the calculating co-occurence matrix
                glcm270=self.calculate_co_occurence_matrix(neighbourhoodMatrix)
                ans90=self.angular_second_moment(glcm270)
                ans1_90=self.entropy_np(glcm270)
                ans2_90=self.con_arr(glcm270)
                idm_sum = self.idm(glcm270)
                # print(list(ans2_90).shape)
                # b = [ans90,ans1_90,ans2_90]
                # a = np.array(a)
                # outputbandCol90.append(b)
                output.append(ans90)
                output_ans1.append(ans1_90)
                output_ans2.append(ans2_90)
                output_idm.append(idm_sum)
            # print(len(outputbandCol90))
            # outputband90.append(outputbandCol90)
        output = output/(1 + np.max(output)-np.min(output))
        
        output = output.reshape(output.shape[0],1)
        print(output.shape)
        output_ans1 = output_ans1/(1 + np.max(output_ans1)-np.min(output_ans1))
        output_ans1 = output_ans1.reshape(output.shape[0],1)
        
        output_ans2 = output_ans2/(1 + np.max(output_ans2)-np.min(output_ans2))
        output_ans2 = output_ans2.reshape(output.shape[0],1)

        output_idm = output_idm/(1 + np.max(output_idm)-np.min(output_idm))
        output_idm = output_idm.reshape(output_idm.shape[0],1)
        
        asm_image = output.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        entropy_image = output_ans1.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))  
        contrast_image = output_ans2.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        idm_image = output_idm.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))

        
        plt.subplot(2, 2, 1)
        plt.imshow(asm_image)
        plt.title("ASM")
        
        plt.subplot(2, 2, 2)
        plt.imshow(entropy_image)
        plt.title("Entropy")
        
        plt.subplot(2, 2, 3)
        plt.imshow(contrast_image)
        plt.title("Contrast")
        
        plt.subplot(2, 2, 4)
        plt.imshow(idm_image)
        plt.title("IDM")
        
        plt.suptitle("GLCM Features 270"+u"\N{DEGREE SIGN}")
        plt.savefig("GLCM Features 270 degrees")

        
        plt.show()
        # outputband90 = np.array(outputband90)
        
        # print()
        # outputband90 = outputband90 * 255/np.amax(outputband90)
        #         print(np.amax(outputband))
        # output_final.append([output.tolist(),output_ans1.tolist(),output_ans2.tolist()])
        for i in range(output.shape[0]):
            x = [output[i], output_ans1[i],output_ans2[i], output_idm[i]]
            output_final_270.append(x)
        # output_final.append(output)
        # output_final.append(output_ans1)
        # output_final.append(output_ans2)
        print(output[0],output_ans1[0],output_ans2[0], output_idm[0])
        # print(output_final)
        
        
        return np.array(output_final_270)
                
                

    def Input_filter_size(self,img_array,filter_size):
        
        self.filter_size = filter_size
        print("You have entered: "+str(self.filter_size))
        outputband270=self.convolution(img_array)
        
        print(outputband270.shape)
        
#         print("0 degree")

        
        # plt.subplot(2, 2, 2)
        # plt.imshow(outputband90)
        # plt.title("90 degree")
        

        return outputband270

"""### 0 Degree GLCM"""

class GLCM_FeatureExtraction_0:
    
    def __init__(self,image,filter_size=None):
        self.img_array=image
        self.filter_size=filter_size
        self.clustered_image=None
    
    def imagePadding(self,band,paddingValue):
        ##vertical Top padding
        ##vertical bottom padding
        band =  np.insert(band,len(band),band[-1]*np.ones(paddingValue)[:,None],axis = 0)
        ##Horizontal left padding
        band = np.insert(band, 0, band[:,0]*np.ones(paddingValue)[:,None], axis=1)
        ##Horizontal right padding
        band = np.insert(band, len(band[0]), band[:,-1]*np.ones(paddingValue)[:,None], axis=1)
        return band
    
    def angular_second_moment(self,numarray):
#         print(type(numarray))
        arr = numarray**2.0
    #     sum of numpy array
        arsum = np.sum(arr)
        # print(a)
        return arsum
    
    # assuming numpy array

    def entropy_np(self,num_array):
        numlog = np.log(1/(num_array+0.0001))
        entrnp = num_array*numlog
        entrnp_sum = np.sum(entrnp)
        return entrnp_sum
    # contrast

    def con_arr(self,input_matrix):
        con_sum = np.sum(a*input_matrix)
        return con_sum
    
    def idm(self,input_matrix):
        idm_sum = np.sum(b*input_matrix)
        return idm_sum
        
    
    def transpose_addition(self, temp):
        temp_arry0=(np.array(temp)).transpose()
        temp_arry0 = temp + temp_arry0
        # sum = np.sum(temp0)
        temp_arry0 = sklearn.preprocessing.normalize(temp_arry0,norm='l2')  
#         print(type(temp_arry0))
#         print(temp_arry0)
        return temp_arry0
    
    def calculate_co_occurence_matrix(self,neighbourhoodMatrix):
        
        #create all the required temp arrays:
        
        rows=len(neighbourhoodMatrix)
        cols=len(neighbourhoodMatrix[0])
        #print(rows)
        #print(cols)
        #neighbourhoodMatrix=[[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]]
        temp0   = np.zeros((256, 256))
        
        
        #calculate the 90 degree co-occurence matrix:
        
        #calculate the 180 degree co-occurence matrix:
        #calculate the 135 degree co-occurence matrix:
        #calculate the 0 degree co-occurence matrix:
        for i in range(rows):
            for j in range(cols):
                if(j+1<cols):
                    ind1=neighbourhoodMatrix[i][j]
                    ind2=neighbourhoodMatrix[i][j+1]
                    temp0[ind1][ind2]=temp0[ind1][ind2]+1
        # glcm90 = self.transpose_addition(temp90)
        glcm0 = temp0
        return glcm0
    


    
    def convolution(self,band):
        if (self.filter_size % 2 == 0):
            raise Exception("K should be odd number.")
            
        # paddingValue = int((self.filter_size - 1)/2)
        paddingValue = 0
        imageBand = self.imagePadding(band,paddingValue)
        arrayshape = imageBand.shape
        outputband90 = []
        output = []
        output_ans1 = []
        output_ans2 = []
        output_idm = []
        output_final_0 = []
        for rowNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
            outputbandCol90 = []

            for colNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
                neighbourhoodMatrix = imageBand[rowNo:rowNo + self.filter_size,colNo:colNo + self.filter_size]
                ## writing code for the calculating co-occurence matrix
                glcm0=self.calculate_co_occurence_matrix(neighbourhoodMatrix)
                ans90=self.angular_second_moment(glcm0)
                ans1_90=self.entropy_np(glcm0)
                ans2_90=self.con_arr(glcm0)
                idm_sum = self.idm(glcm0)
                # print(list(ans2_90).shape)
                # b = [ans90,ans1_90,ans2_90]
                # a = np.array(a)
                # outputbandCol90.append(b)
                output.append(ans90)
                output_ans1.append(ans1_90)
                output_ans2.append(ans2_90)
                output_idm.append(idm_sum)
            # print(len(outputbandCol90))
            # outputband90.append(outputbandCol90)
        output = output/(1 + np.max(output)-np.min(output))
        output = output.reshape(output.shape[0],1)
        print(output.shape)
        output_ans1 = output_ans1/(1+ np.max(output_ans1)-np.min(output_ans1))
        output_ans1 = output_ans1.reshape(output.shape[0],1)
        
        output_ans2 = output_ans2/(1 + np.max(output_ans2)-np.min(output_ans2))
        output_ans2 = output_ans2.reshape(output.shape[0],1)

        output_idm = output_idm/(1 + np.max(output_idm)-np.min(output_idm))
        output_idm = output_idm.reshape(output_idm.shape[0],1)
        
        asm_image = output.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        entropy_image = output_ans1.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))  
        contrast_image = output_ans2.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        idm_image = output_idm.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))

        
        plt.subplot(2, 2, 1)
        plt.imshow(asm_image)
        plt.title("ASM")
        
        plt.subplot(2, 2, 2)
        plt.imshow(entropy_image)
        plt.title("Entropy")
        
        plt.subplot(2, 2, 3)
        plt.imshow(contrast_image)
        plt.title("Contrast")
        
        plt.subplot(2, 2, 4)
        plt.imshow(idm_image)
        plt.title("IDM")
        
        plt.suptitle("GLCM Features 0"+u"\N{DEGREE SIGN}")
        plt.savefig("GLCM Features 0 degrees")

        
        plt.show()
        # outputband90 = np.array(outputband90)
        
        # print()
        # outputband90 = outputband90 * 255/np.amax(outputband90)
        #         print(np.amax(outputband))
        # output_final.append([output.tolist(),output_ans1.tolist(),output_ans2.tolist()])
        for i in range(output.shape[0]):
            x = [output[i], output_ans1[i],output_ans2[i], output_idm[i]]
            output_final_0.append(x)
        # output_final.append(output)
        # output_final.append(output_ans1)
        # output_final.append(output_ans2)
        print(output[0],output_ans1[0],output_ans2[0],output_idm[0])
        # print(output_final)
        
        
        return np.array(output_final_0)
                
                

    def Input_filter_size(self,img_array,filter_size):
        
        self.filter_size = filter_size
        print("You have entered: "+str(self.filter_size))
        outputband0=self.convolution(img_array)
        
        print(outputband0.shape)
        
#       print("0 degree")

        
        # plt.subplot(2, 2, 2)
        # plt.imshow(outputband90)
        # plt.title("90 degree")
        

        return outputband0

"""### 45 Degree GLCM"""

class GLCM_FeatureExtraction_45:
    
    def __init__(self,image,filter_size=None):
        self.img_array=image
        self.filter_size=filter_size
        self.clustered_image=None
    
    def imagePadding(self,band,paddingValue):
        ##vertical Top padding
        ##vertical bottom padding
        band =  np.insert(band,len(band),band[-1]*np.ones(paddingValue)[:,None],axis = 0)
        ##Horizontal left padding
        band = np.insert(band, 0, band[:,0]*np.ones(paddingValue)[:,None], axis=1)
        ##Horizontal right padding
        band = np.insert(band, len(band[0]), band[:,-1]*np.ones(paddingValue)[:,None], axis=1)
        return band
    
    def angular_second_moment(self,numarray):
#         print(type(numarray))
        arr = numarray**2.0
    #     sum of numpy array
        arsum = np.sum(arr)
        # print(a)
        return arsum
    
    # assuming numpy array

    def entropy_np(self,num_array):
        numlog = np.log(1/(num_array+0.0001))
        entrnp = num_array*numlog
        entrnp_sum = np.sum(entrnp)
        return entrnp_sum
    # contrast

    def con_arr(self,input_matrix):
        con_sum = np.sum(a*input_matrix)
        return con_sum
    
    def idm(self,input_matrix):
        idm_sum = np.sum(b*input_matrix)
        return idm_sum
        
    
    def transpose_addition(self, temp):
        temp_arry0=(np.array(temp)).transpose()
        temp_arry0 = temp + temp_arry0
        # sum = np.sum(temp0)
        temp_arry0 = sklearn.preprocessing.normalize(temp_arry0,norm='l2')  
#         print(type(temp_arry0))
#         print(temp_arry0)
        return temp_arry0
    # @numba.jit(nopython=True)
    def calculate_co_occurence_matrix(self,neighbourhoodMatrix):
        
        #create all the required temp arrays:
        
        rows=len(neighbourhoodMatrix)
        cols=len(neighbourhoodMatrix[0])
        #print(rows)
        #print(cols)
        #neighbourhoodMatrix=[[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]]
        temp45   = np.zeros((256, 256))
        
        
        #calculate the 90 degree co-occurence matrix:
        
        #calculate the 45 degree co-occurence matrix:
        for i in range(rows):
            for j in range(cols):
                if(i-1 >=0 and j+1<cols):
                    ind1=neighbourhoodMatrix[i][j]
                    ind2=neighbourhoodMatrix[i-1][j+1]
                    temp45[ind1][ind2]=temp45[ind1][ind2]+1
        # glcm90 = self.transpose_addition(temp90)
        glcm45 = temp45
        return glcm45
    


    
    def convolution(self,band):
        if (self.filter_size % 2 == 0):
            raise Exception("K should be odd number.")
            
        # paddingValue = int((self.filter_size - 1)/2)
        paddingValue = 0
        imageBand = self.imagePadding(band,paddingValue)
        arrayshape = imageBand.shape
        outputband90 = []
        output = []
        output_ans1 = []
        output_ans2 = []
        output_idm = []
        output_final_45 = []
        for rowNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
            outputbandCol90 = []

            for colNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
                neighbourhoodMatrix = imageBand[rowNo:rowNo + self.filter_size,colNo:colNo + self.filter_size]
                ## writing code for the calculating co-occurence matrix
                glcm45=self.calculate_co_occurence_matrix(neighbourhoodMatrix)
                ans90=self.angular_second_moment(glcm45)
                ans1_90=self.entropy_np(glcm45)
                ans2_90=self.con_arr(glcm45)
                idm_sum = self.idm(glcm45)
                # print(list(ans2_90).shape)
                # b = [ans90,ans1_90,ans2_90]
                # a = np.array(a)
                # outputbandCol90.append(b)
                output.append(ans90)
                output_ans1.append(ans1_90)
                output_ans2.append(ans2_90)
                output_idm.append(idm_sum)
            # print(len(outputbandCol90))
            # outputband90.append(outputbandCol90)
        output = output/(1 + np.max(output)-np.min(output))
        output = output.reshape(output.shape[0],1)
        print(output.shape)
        output_ans1 = output_ans1/(1 + np.max(output_ans1)-np.min(output_ans1))
        output_ans1 = output_ans1.reshape(output.shape[0],1)
        
        output_ans2 = output_ans2/(1 + np.max(output_ans2)-np.min(output_ans2))
        output_ans2 = output_ans2.reshape(output.shape[0],1)

        output_idm = output_idm/(1 + np.max(output_idm)-np.min(output_idm))
        output_idm = output_idm.reshape(output_idm.shape[0],1)
        
        asm_image = output.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        entropy_image = output_ans1.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))  
        contrast_image = output_ans2.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        idm_image = output_idm.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))

        
        plt.subplot(2, 2, 1)
        plt.imshow(asm_image)
        plt.title("ASM")
        
        plt.subplot(2, 2, 2)
        plt.imshow(entropy_image)
        plt.title("Entropy")
        
        plt.subplot(2, 2, 3)
        plt.imshow(contrast_image)
        plt.title("Contrast")
        
        plt.subplot(2, 2, 4)
        plt.imshow(idm_image)
        plt.title("IDM")
        
        plt.suptitle("GLCM Features 45"+u"\N{DEGREE SIGN}")
        plt.savefig("GLCM Features 45 degrees")

        
        plt.show()
        # outputband90 = np.array(outputband90)
        
        # print()
        # outputband90 = outputband90 * 255/np.amax(outputband90)
        #         print(np.amax(outputband))
        # output_final.append([output.tolist(),output_ans1.tolist(),output_ans2.tolist()])
        for i in range(output.shape[0]):
            x = [output[i], output_ans1[i],output_ans2[i], output_idm[i]]
            output_final_45.append(x)
        # output_final.append(output)
        # output_final.append(output_ans1)
        # output_final.append(output_ans2)
        print(output[0],output_ans1[0],output_ans2[0], output_idm[0])
        # print(output_final)
        
        
        return np.array(output_final_45)
                
                

    def Input_filter_size(self,img_array,filter_size):
        
        self.filter_size = filter_size
        print("You have entered: "+str(self.filter_size))
        outputband45=self.convolution(img_array)
        
        print(outputband45.shape)
        
#         print("0 degree")

        
        # plt.subplot(2, 2, 2)
        # plt.imshow(outputband90)
        # plt.title("90 degree")
        

        return outputband45

"""### 90 Degree GLCM"""

class GLCM_FeatureExtraction_90:
    
    def __init__(self,image,filter_size=None):
        self.img_array=image
        self.filter_size=filter_size
        self.clustered_image=None
    
    def imagePadding(self,band,paddingValue):
        ##vertical Top padding
        ##vertical bottom padding
        band =  np.insert(band,len(band),band[-1]*np.ones(paddingValue)[:,None],axis = 0)
        ##Horizontal left padding
        band = np.insert(band, 0, band[:,0]*np.ones(paddingValue)[:,None], axis=1)
        ##Horizontal right padding
        band = np.insert(band, len(band[0]), band[:,-1]*np.ones(paddingValue)[:,None], axis=1)
        return band
    
    def angular_second_moment(self,numarray):
#         print(type(numarray))
        arr = numarray**2.0
    #     sum of numpy array
        arsum = np.sum(arr)
        # print(a)
        return arsum
    
    # assuming numpy array

    def entropy_np(self,num_array):
        numlog = np.log(1/(num_array+0.0001))
        entrnp = num_array*numlog
        entrnp_sum = np.sum(entrnp)
        return entrnp_sum
    # contrast

    def con_arr(self,input_matrix):
        con_sum = np.sum(a*input_matrix)
        return con_sum
    
    def idm(self,input_matrix):
        idm_sum = np.sum(b*input_matrix)
        return idm_sum
        
        
    
    def transpose_addition(self, temp):
        temp_arry0=(np.array(temp)).transpose()
        temp_arry0 = temp + temp_arry0
        # sum = np.sum(temp0)
        temp_arry0 = sklearn.preprocessing.normalize(temp_arry0,norm='l2')  
#         print(type(temp_arry0))
#         print(temp_arry0)
        return temp_arry0
    
    def calculate_co_occurence_matrix(self,neighbourhoodMatrix):
        
        #create all the required temp arrays:
        
        rows=len(neighbourhoodMatrix)
        cols=len(neighbourhoodMatrix[0])
        #print(rows)
        #print(cols)
        #neighbourhoodMatrix=[[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]]
        temp90   = np.zeros((256, 256))
        
        
        #calculate the 90 degree co-occurence matrix:
        
        #calculate the 90 degree co-occurence matrix:
        for i in range(rows):
            for j in range(cols):
                if(i-1>=0):
                    ind1=neighbourhoodMatrix[i][j]
                    ind2=neighbourhoodMatrix[i-1][j]
                    temp90[ind1][ind2]=temp90[ind1][ind2]+1
        # glcm90 = self.transpose_addition(temp90)
        glcm90 = temp90
        return glcm90
    


    
    def convolution(self,band):
        if (self.filter_size % 2 == 0):
            raise Exception("K should be odd number.")
            
        # paddingValue = int((self.filter_size - 1)/2)
        paddingValue = 0
        imageBand = self.imagePadding(band,paddingValue)
        arrayshape = imageBand.shape
        outputband90 = []
        output = []
        output_ans1 = []
        output_ans2 = []
        output_idm = []
        output_final_90 = []
        for rowNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
            outputbandCol90 = []

            for colNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
                neighbourhoodMatrix = imageBand[rowNo:rowNo + self.filter_size,colNo:colNo + self.filter_size]
                ## writing code for the calculating co-occurence matrix
                glcm90=self.calculate_co_occurence_matrix(neighbourhoodMatrix)
                ans90=self.angular_second_moment(glcm90)
                ans1_90=self.entropy_np(glcm90)
                ans2_90=self.con_arr(glcm90)
                idm_sum = self.idm(glcm90)
                # print(list(ans2_90).shape)
                # b = [ans90,ans1_90,ans2_90]
                # a = np.array(a)
                # outputbandCol90.append(b)
                output.append(ans90)
                output_ans1.append(ans1_90)
                output_ans2.append(ans2_90)
                output_idm.append(idm_sum)
            # print(len(outputbandCol90))
            # outputband90.append(outputbandCol90)
        output = output/(1 + np.max(output)-np.min(output))
        output = output.reshape(output.shape[0],1)
        print(output.shape)
        output_ans1 = output_ans1/(1 + np.max(output_ans1)-np.min(output_ans1))
        output_ans1 = output_ans1.reshape(output.shape[0],1)
        
        output_ans2 = output_ans2/(1 + np.max(output_ans2)-np.min(output_ans2))
        output_ans2 = output_ans2.reshape(output.shape[0],1)

        output_idm = output_idm/(1 + np.max(output_idm)-np.min(output_idm))
        output_idm = output_idm.reshape(output_idm.shape[0],1)
        
        asm_image = output.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        entropy_image = output_ans1.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))  
        contrast_image = output_ans2.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        idm_image = output_idm.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))

        
        plt.subplot(2, 2, 1)
        plt.imshow(asm_image)
        plt.title("ASM")
        
        plt.subplot(2, 2, 2)
        plt.imshow(entropy_image)
        plt.title("Entropy")
        
        plt.subplot(2, 2, 3)
        plt.imshow(contrast_image)
        plt.title("Contrast")
        
        plt.subplot(2, 2, 4)
        plt.imshow(idm_image)
        plt.title("IDM")
        
        plt.suptitle("GLCM Features 90"+u"\N{DEGREE SIGN}")
        plt.savefig("GLCM Features 90 degrees")

        
        plt.show()
        # outputband90 = np.array(outputband90)
        
        # print()
        # outputband90 = outputband90 * 255/np.amax(outputband90)
        #         print(np.amax(outputband))
        # output_final.append([output.tolist(),output_ans1.tolist(),output_ans2.tolist()])
        for i in range(output.shape[0]):
            x = [output[i], output_ans1[i],output_ans2[i], output_idm[i]]
            output_final_90.append(x)
        # output_final.append(output)
        # output_final.append(output_ans1)
        # output_final.append(output_ans2)
        print(output[0],output_ans1[0],output_ans2[0], output_idm[0])
        # print(output_final)
        
        
        return np.array(output_final_90)
                
                

    def Input_filter_size(self,img_array,filter_size):
        
        self.filter_size = filter_size
        print("You have entered: "+str(self.filter_size))
        outputband90=self.convolution(img_array)
        
        print(outputband90.shape)
        
#         print("0 degree")

        
        # plt.subplot(2, 2, 2)
        # plt.imshow(outputband90)
        # plt.title("90 degree")
        

        return outputband90

"""### 135 Degree GLCM"""

class GLCM_FeatureExtraction_135:
    
    def __init__(self,image,filter_size=None):
        self.img_array=image
        self.filter_size=filter_size
        self.clustered_image=None
    
    def imagePadding(self,band,paddingValue):
        ##vertical Top padding
        ##vertical bottom padding
        band =  np.insert(band,len(band),band[-1]*np.ones(paddingValue)[:,None],axis = 0)
        ##Horizontal left padding
        band = np.insert(band, 0, band[:,0]*np.ones(paddingValue)[:,None], axis=1)
        ##Horizontal right padding
        band = np.insert(band, len(band[0]), band[:,-1]*np.ones(paddingValue)[:,None], axis=1)
        return band
    
    def angular_second_moment(self,numarray):
#         print(type(numarray))
        arr = numarray**2.0
    #     sum of numpy array
        arsum = np.sum(arr)
        # print(a)
        return arsum
    
    # assuming numpy array

    def entropy_np(self,num_array):
        numlog = np.log(1/(num_array+0.0001))
        entrnp = num_array*numlog
        entrnp_sum = np.sum(entrnp)
        return entrnp_sum
    # contrast

    def con_arr(self,input_matrix):
        con_sum = np.sum(a*input_matrix)
        return con_sum
        
    def idm(self,input_matrix):
        idm_sum = np.sum(b*input_matrix)
        return idm_sum
    
    def transpose_addition(self, temp):
        temp_arry0=(np.array(temp)).transpose()
        temp_arry0 = temp + temp_arry0
        # sum = np.sum(temp0)
        temp_arry0 = sklearn.preprocessing.normalize(temp_arry0,norm='l2')  
#         print(type(temp_arry0))
#         print(temp_arry0)
        return temp_arry0
    
    def calculate_co_occurence_matrix(self,neighbourhoodMatrix):
        
        #create all the required temp arrays:
        
        rows=len(neighbourhoodMatrix)
        cols=len(neighbourhoodMatrix[0])
        #print(rows)
        #print(cols)
        #neighbourhoodMatrix=[[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]]
        temp135   = np.zeros((256, 256))
        
        
        #calculate the 0 degree co-occurence matrix:
        for i in range(rows):
            for j in range(cols):
                if(i-1 >=0 and j-1>=0):
                    ind1=neighbourhoodMatrix[i][j]
                    ind2=neighbourhoodMatrix[i-1][j-1]
                    temp135[ind1][ind2]=temp135[ind1][ind2]+1
        # glcm90 = self.transpose_addition(temp90)
        glcm135 = temp135
        return glcm135
    


    
    def convolution(self,band):
        if (self.filter_size % 2 == 0):
            raise Exception("K should be odd number.")
            
        # paddingValue = int((self.filter_size - 1)/2)
        paddingValue = 0
        imageBand = self.imagePadding(band,paddingValue)
        arrayshape = imageBand.shape
        outputband90 = []
        output = []
        output_ans1 = []
        output_ans2 = []
        output_idm = []
        output_final_135 = []
        for rowNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
            outputbandCol90 = []

            for colNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
                neighbourhoodMatrix = imageBand[rowNo:rowNo + self.filter_size,colNo:colNo + self.filter_size]
                ## writing code for the calculating co-occurence matrix
                glcm135=self.calculate_co_occurence_matrix(neighbourhoodMatrix)
                ans90=self.angular_second_moment(glcm135)
                ans1_90=self.entropy_np(glcm135)
                ans2_90=self.con_arr(glcm135)
                idm_sum = self.idm(glcm135)
                # print(list(ans2_90).shape)
                # b = [ans90,ans1_90,ans2_90]
                # a = np.array(a)
                # outputbandCol90.append(b)
                output.append(ans90)
                output_ans1.append(ans1_90)
                output_ans2.append(ans2_90)
                output_idm.append(idm_sum)
            # print(len(outputbandCol90))
            # outputband90.append(outputbandCol90)
        output = output/(1 + np.max(output)-np.min(output))
        output = output.reshape(output.shape[0],1)
        print(output.shape)
        output_ans1 = output_ans1/(1 + np.max(output_ans1)-np.min(output_ans1))
        output_ans1 = output_ans1.reshape(output.shape[0],1)
        
        output_ans2 = output_ans2/(1 + np.max(output_ans2)-np.min(output_ans2))
        output_ans2 = output_ans2.reshape(output.shape[0],1)

        output_idm = output_idm/(1 + np.max(output_idm)-np.min(output_idm))
        output_idm = output_idm.reshape(output_idm.shape[0],1)
        
        asm_image = output.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        entropy_image = output_ans1.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))  
        contrast_image = output_ans2.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        idm_image = output_idm.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))

        
        plt.subplot(2, 2, 1)
        plt.imshow(asm_image)
        plt.title("ASM")
        
        plt.subplot(2, 2, 2)
        plt.imshow(entropy_image)
        plt.title("Entropy")
        
        plt.subplot(2, 2, 3)
        plt.imshow(contrast_image)
        plt.title("Contrast")
        
        plt.subplot(2, 2, 4)
        plt.imshow(idm_image)
        plt.title("IDM")
        
        plt.suptitle("GLCM Features 135"+u"\N{DEGREE SIGN}")
        plt.savefig("GLCM Features 135 degrees")

        
        plt.show()
        # outputband90 = np.array(outputband90)
        
        # print()
        # outputband90 = outputband90 * 255/np.amax(outputband90)
        #         print(np.amax(outputband))
        # output_final.append([output.tolist(),output_ans1.tolist(),output_ans2.tolist()])
        for i in range(output.shape[0]):
            x = [output[i], output_ans1[i],output_ans2[i], output_idm[i]]
            output_final_135.append(x)
        # output_final.append(output)
        # output_final.append(output_ans1)
        # output_final.append(output_ans2)
        print(output[0],output_ans1[0],output_ans2[0], output_idm[0])
        # print(output_final)
        
        
        return np.array(output_final_135)
                
                

    def Input_filter_size(self,img_array,filter_size):
        
        self.filter_size = filter_size
        print("You have entered: "+str(self.filter_size))
        outputband135=self.convolution(img_array)
        
        print(outputband135.shape)
        
#         print("0 degree")

        
        # plt.subplot(2, 2, 2)
        # plt.imshow(outputband90)
        # plt.title("90 degree")
        

        return outputband135

"""### 180 Degree GLCM"""

class GLCM_FeatureExtraction_180:
    
    def __init__(self,image,filter_size=None):
        self.img_array=image
        self.filter_size=filter_size
        self.clustered_image=None
    
    def imagePadding(self,band,paddingValue):
        ##vertical Top padding
        ##vertical bottom padding
        band =  np.insert(band,len(band),band[-1]*np.ones(paddingValue)[:,None],axis = 0)
        ##Horizontal left padding
        band = np.insert(band, 0, band[:,0]*np.ones(paddingValue)[:,None], axis=1)
        ##Horizontal right padding
        band = np.insert(band, len(band[0]), band[:,-1]*np.ones(paddingValue)[:,None], axis=1)
        return band
    
    def angular_second_moment(self,numarray):
#         print(type(numarray))
        arr = numarray**2.0
    #     sum of numpy array
        arsum = np.sum(arr)
        # print(a)
        return arsum
    
    # assuming numpy array

    def entropy_np(self,num_array):
        numlog = np.log(1/(num_array+0.0001))
        entrnp = num_array*numlog
        entrnp_sum = np.sum(entrnp)
        return entrnp_sum
    # contrast

    def con_arr(self,input_matrix):
        con_sum = np.sum(a*input_matrix)
        return con_sum
        
    def idm(self,input_matrix):
        idm_sum = np.sum(b*input_matrix)
        return idm_sum
    
    def transpose_addition(self, temp):
        temp_arry0=(np.array(temp)).transpose()
        temp_arry0 = temp + temp_arry0
        # sum = np.sum(temp0)
        temp_arry0 = sklearn.preprocessing.normalize(temp_arry0,norm='l2')  
#         print(type(temp_arry0))
#         print(temp_arry0)
        return temp_arry0
    
    def calculate_co_occurence_matrix(self,neighbourhoodMatrix):
        
        #create all the required temp arrays:
        
        rows=len(neighbourhoodMatrix)
        cols=len(neighbourhoodMatrix[0])
        #print(rows)
        #print(cols)
        #neighbourhoodMatrix=[[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]]
        temp180   = np.zeros((256, 256))
        
        
        #calculate the 90 degree co-occurence matrix:
        
        #calculate the 180 degree co-occurence matrix:
        for i in range(rows):
            for j in range(cols):
                if(j-1>=0):
                    ind1=neighbourhoodMatrix[i][j]
                    ind2=neighbourhoodMatrix[i][j-1]
                    temp180[ind1][ind2]=temp180[ind1][ind2]+1
        # glcm90 = self.transpose_addition(temp90)
        glcm180 = temp180
        return glcm180
    


    
    def convolution(self,band):
        if (self.filter_size % 2 == 0):
            raise Exception("K should be odd number.")
            
        # paddingValue = int((self.filter_size - 1)/2)
        paddingValue = 0
        imageBand = self.imagePadding(band,paddingValue)
        arrayshape = imageBand.shape
        outputband90 = []
        output = []
        output_ans1 = []
        output_ans2 = []
        output_idm = []
        output_final_180 = []
        for rowNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
            outputbandCol90 = []

            for colNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
                neighbourhoodMatrix = imageBand[rowNo:rowNo + self.filter_size,colNo:colNo + self.filter_size]
                ## writing code for the calculating co-occurence matrix
                glcm180=self.calculate_co_occurence_matrix(neighbourhoodMatrix)
                ans90=self.angular_second_moment(glcm180)
                ans1_90=self.entropy_np(glcm180)
                ans2_90=self.con_arr(glcm180)
                idm_sum = self.idm(glcm180)
                # print(list(ans2_90).shape)
                # b = [ans90,ans1_90,ans2_90]
                # a = np.array(a)
                # outputbandCol90.append(b)
                output.append(ans90)
                output_ans1.append(ans1_90)
                output_ans2.append(ans2_90)
                output_idm.append(idm_sum)
            # print(len(outputbandCol90))
            # outputband90.append(outputbandCol90)
        output = output/(1 + np.max(output)-np.min(output))
        output = output.reshape(output.shape[0],1)
        print(output.shape)
        output_ans1 = output_ans1/(1 + np.max(output_ans1)-np.min(output_ans1))
        output_ans1 = output_ans1.reshape(output.shape[0],1)
        
        output_ans2 = output_ans2/(1 + np.max(output_ans2)-np.min(output_ans2))
        output_ans2 = output_ans2.reshape(output.shape[0],1)

        output_idm = output_idm/(1 + np.max(output_idm)-np.min(output_idm))
        output_idm = output_idm.reshape(output_idm.shape[0],1)
        
        asm_image = output.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        entropy_image = output_ans1.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))  
        contrast_image = output_ans2.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        idm_image = output_idm.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))

        
        plt.subplot(2, 2, 1)
        plt.imshow(asm_image)
        plt.title("ASM")
        
        plt.subplot(2, 2, 2)
        plt.imshow(entropy_image)
        plt.title("Entropy")
        
        plt.subplot(2, 2, 3)
        plt.imshow(contrast_image)
        plt.title("Contrast")
        
        plt.subplot(2, 2, 4)
        plt.imshow(idm_image)
        plt.title("IDM")
        
        plt.suptitle("GLCM Features 180"+u"\N{DEGREE SIGN}")
        plt.savefig("GLCM Features 180 degrees")

        
        plt.show()
        # outputband90 = np.array(outputband90)
        
        # print()
        # outputband90 = outputband90 * 255/np.amax(outputband90)
        #         print(np.amax(outputband))
        # output_final.append([output.tolist(),output_ans1.tolist(),output_ans2.tolist()])
        for i in range(output.shape[0]):
            x = [output[i], output_ans1[i],output_ans2[i], output_idm[i]]
            output_final_180.append(x)
        # output_final.append(output)
        # output_final.append(output_ans1)
        # output_final.append(output_ans2)
        print(output[0],output_ans1[0],output_ans2[0], output_idm[0])
        # print(output_final)
        
        
        return np.array(output_final_180)
                
                

    def Input_filter_size(self,img_array,filter_size):
        
        self.filter_size = filter_size
        print("You have entered: "+str(self.filter_size))
        outputband180=self.convolution(img_array)
        
        print(outputband180.shape)
        
#         print("0 degree")

        
        # plt.subplot(2, 2, 2)
        # plt.imshow(outputband90)
        # plt.title("90 degree")
        

        return outputband180




"""####**225** Degree GLCM"""

class GLCM_FeatureExtraction_225:
    
    def __init__(self,image,filter_size=None):
        self.img_array=image
        self.filter_size=filter_size
        self.clustered_image=None
    
    def imagePadding(self,band,paddingValue):
        ##vertical Top padding
        ##vertical bottom padding
        band =  np.insert(band,len(band),band[-1]*np.ones(paddingValue)[:,None],axis = 0)
        ##Horizontal left padding
        band = np.insert(band, 0, band[:,0]*np.ones(paddingValue)[:,None], axis=1)
        ##Horizontal right padding
        band = np.insert(band, len(band[0]), band[:,-1]*np.ones(paddingValue)[:,None], axis=1)
        return band
    
    def angular_second_moment(self,numarray):
#         print(type(numarray))
        arr = numarray**2.0
    #     sum of numpy array
        arsum = np.sum(arr)
        # print(a)
        return arsum
    
    # assuming numpy array

    def entropy_np(self,num_array):
        numlog = np.log(1/(num_array+0.0001))
        entrnp = num_array*numlog
        entrnp_sum = np.sum(entrnp)
        return entrnp_sum
    # contrast

    def con_arr(self,input_matrix):
        con_sum = np.sum(a*input_matrix)
        return con_sum

    def idm(self,input_matrix):
        idm_sum = np.sum(b*input_matrix)
        return idm_sum
        
    
    def transpose_addition(self, temp):
        temp_arry0=(np.array(temp)).transpose()
        temp_arry0 = temp + temp_arry0
        # sum = np.sum(temp0)
        temp_arry0 = sklearn.preprocessing.normalize(temp_arry0,norm='l2')  
#         print(type(temp_arry0))
#         print(temp_arry0)
        return temp_arry0
    
    def calculate_co_occurence_matrix(self,neighbourhoodMatrix):
        
        #create all the required temp arrays:
        
        rows=len(neighbourhoodMatrix)
        cols=len(neighbourhoodMatrix[0])
        #print(rows)
        #print(cols)
        #neighbourhoodMatrix=[[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]]
        temp225   = np.zeros((256, 256))
        
        
        #calculate the 90 degree co-occurence matrix:
        
        #calculate the 180 degree co-occurence matrix:
        for i in range(rows):
            for j in range(cols):
                if(j-1>=0):
                    ind1=neighbourhoodMatrix[i][j]
                    ind2=neighbourhoodMatrix[i][j-1]
                    temp225[ind1][ind2]=temp225[ind1][ind2]+1
        # glcm90 = self.transpose_addition(temp90)
        glcm225 = temp225
        return glcm225
    


    
    def convolution(self,band):
        if (self.filter_size % 2 == 0):
            raise Exception("K should be odd number.")
            
        paddingValue = 0
        imageBand = self.imagePadding(band,paddingValue)
        arrayshape = imageBand.shape
        print(arrayshape)
        outputband90 = []
        output = []
        output_ans1 = []
        output_ans2 = []
        output_idm = []
        output_final_225 = []
        for rowNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
            outputbandCol90 = []

            for colNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
                neighbourhoodMatrix = imageBand[rowNo:rowNo + self.filter_size,colNo:colNo + self.filter_size]
                ## writing code for the calculating co-occurence matrix
                glcm225=self.calculate_co_occurence_matrix(neighbourhoodMatrix)
                ans90=self.angular_second_moment(glcm225)
                ans1_90=self.entropy_np(glcm225)
                ans2_90=self.con_arr(glcm225)
                idm_sum = self.idm(glcm225)
                # print(list(ans2_90).shape)
                # b = [ans90,ans1_90,ans2_90]
                # a = np.array(a)
                # outputbandCol90.append(b)
                output.append(ans90)
                output_ans1.append(ans1_90)
                output_ans2.append(ans2_90)
                output_idm.append(idm_sum)
            # print(len(outputbandCol90))
            # outputband90.append(outputbandCol90)
        output = output/(1 + np.max(output)-np.min(output))
        output = output.reshape(output.shape[0],1)
        print(output.shape)
        output_ans1 = output_ans1/(1 + np.max(output_ans1)-np.min(output_ans1))
        output_ans1 = output_ans1.reshape(output.shape[0],1)
        
        output_ans2 = output_ans2/(1 + np.max(output_ans2)-np.min(output_ans2))
        output_ans2 = output_ans2.reshape(output.shape[0],1)

        output_idm = output_idm/(1 + np.max(output_idm)-np.min(output_idm))
        output_idm = output_idm.reshape(output_idm.shape[0],1)

        asm_image = output.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        entropy_image = output_ans1.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))  
        contrast_image = output_ans2.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        idm_image = output_idm.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))

        
        plt.subplot(2, 2, 1)
        plt.imshow(asm_image)
        plt.title("ASM")
        
        plt.subplot(2, 2, 2)
        plt.imshow(entropy_image)
        plt.title("Entropy")
        
        plt.subplot(2, 2, 3)
        plt.imshow(contrast_image)
        plt.title("Contrast")
        
        plt.subplot(2, 2, 4)
        plt.imshow(idm_image)
        plt.title("IDM")
        
        plt.suptitle("GLCM Features 225"+u"\N{DEGREE SIGN}")
        plt.savefig("GLCM Features 225 degrees")

        
        plt.show()
        # outputband90 = np.array(outputband90)
        
        # print()
        # outputband90 = outputband90 * 255/np.amax(outputband90)
        #         print(np.amax(outputband))
        # output_final.append([output.tolist(),output_ans1.tolist(),output_ans2.tolist()])
        for i in range(output.shape[0]):
            x = [output[i], output_ans1[i],output_ans2[i], output_idm[i]]
            output_final_225.append(x)
        # output_final.append(output)
        # output_final.append(output_ans1)
        # output_final.append(output_ans2)
        print(output[0],output_ans1[0],output_ans2[0], output_idm[0])
        # print(output_final)
        
        
        return np.array(output_final_225)
                
                

    def Input_filter_size(self,img_array,filter_size):
        
        self.filter_size = filter_size
        print("You have entered: "+str(self.filter_size))
        outputband225=self.convolution(img_array)
        
        print(outputband225.shape)
        
#         print("0 degree")

        
        # plt.subplot(2, 2, 2)
        # plt.imshow(outputband90)
        # plt.title("90 degree")
        

        return outputband225



"""##315 Degree GLCM"""

class GLCM_FeatureExtraction_315:
    
    def __init__(self,image,filter_size=None):
        self.img_array=image
        self.filter_size=filter_size
        self.clustered_image=None
    
    def imagePadding(self,band,paddingValue):
        ##vertical Top padding
        ##vertical bottom padding
        band =  np.insert(band,len(band),band[-1]*np.ones(paddingValue)[:,None],axis = 0)
        ##Horizontal left padding
        band = np.insert(band, 0, band[:,0]*np.ones(paddingValue)[:,None], axis=1)
        ##Horizontal right padding
        band = np.insert(band, len(band[0]), band[:,-1]*np.ones(paddingValue)[:,None], axis=1)
        return band
    
    def angular_second_moment(self,numarray):
#         print(type(numarray))
        arr = numarray**2.0
    #     sum of numpy array
        arsum = np.sum(arr)
        # print(a)
        return arsum
    
    # assuming numpy array

    def entropy_np(self,num_array):
        numlog = np.log(1/(num_array+0.0001))
        entrnp = num_array*numlog
        entrnp_sum = np.sum(entrnp)
        return entrnp_sum
    # contrast

    def con_arr(self,input_matrix):
        con_sum = np.sum(a*input_matrix)
        return con_sum

    def idm(self,input_matrix):
        idm_sum = np.sum(b*input_matrix)
        return idm_sum
        
    
    def transpose_addition(self, temp):
        temp_arry0=(np.array(temp)).transpose()
        temp_arry0 = temp + temp_arry0
        # sum = np.sum(temp0)
        temp_arry0 = sklearn.preprocessing.normalize(temp_arry0,norm='l2')  
#         print(type(temp_arry0))
#         print(temp_arry0)
        return temp_arry0
    
    def calculate_co_occurence_matrix(self,neighbourhoodMatrix):
        
        #create all the required temp arrays:
        
        rows=len(neighbourhoodMatrix)
        cols=len(neighbourhoodMatrix[0])
        #print(rows)
        #print(cols)
        #neighbourhoodMatrix=[[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]]
        temp315   = np.zeros((256, 256))
        
        
        #calculate the 90 degree co-occurence matrix:
        
        #calculate the 180 degree co-occurence matrix:
        for i in range(rows):
            for j in range(cols):
                if(j-1>=0):
                    ind1=neighbourhoodMatrix[i][j]
                    ind2=neighbourhoodMatrix[i][j-1]
                    temp315[ind1][ind2]=temp315[ind1][ind2]+1
        # glcm90 = self.transpose_addition(temp90)
        glcm315 = temp315
        return glcm315
    


    
    def convolution(self,band):
        if (self.filter_size % 2 == 0):
            raise Exception("K should be odd number.")
            
        paddingValue = 0
        imageBand = self.imagePadding(band,paddingValue)
        arrayshape = imageBand.shape
        outputband90 = []
        output = []
        output_ans1 = []
        output_ans2 = []
        output_idm = []
        output_final_315 = []
        for rowNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
            outputbandCol90 = []

            for colNo in range(int((self.filter_size - 1)/2), int(arrayshape[0] - (self.filter_size -1)/2)):#/stride)):
                neighbourhoodMatrix = imageBand[rowNo:rowNo + self.filter_size,colNo:colNo + self.filter_size]
                ## writing code for the calculating co-occurence matrix
                glcm315=self.calculate_co_occurence_matrix(neighbourhoodMatrix)
                ans90=self.angular_second_moment(glcm315)
                ans1_90=self.entropy_np(glcm315)
                ans2_90=self.con_arr(glcm315)
                idm_sum = self.idm(glcm315)
                # print(list(ans2_90).shape)
                # b = [ans90,ans1_90,ans2_90]
                # a = np.array(a)
                # outputbandCol90.append(b)
                output.append(ans90)
                output_ans1.append(ans1_90)
                output_ans2.append(ans2_90)
                output_idm.append(idm_sum)
            # print(len(outputbandCol90))
            # outputband90.append(outputbandCol90)
        output = output/(1 + np.max(output)-np.min(output))
        output = output.reshape(output.shape[0],1)
        print(output.shape)
                        
        output_ans1 = output_ans1/(1 + np.max(output_ans1)-np.min(output_ans1))
        output_ans1 = output_ans1.reshape(output.shape[0],1)
        
            
        output_ans2 = output_ans2/(1 + np.max(output_ans2)-np.min(output_ans2))
        output_ans2 = output_ans2.reshape(output.shape[0],1)

        
        output_idm = output_idm/(1 + np.max(output_idm)-np.min(output_idm))
        output_idm = output_idm.reshape(output_idm.shape[0],1)
        
        asm_image = output.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        entropy_image = output_ans1.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))  
        contrast_image = output_ans2.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))
        idm_image = output_idm.reshape((int(output.shape[0]**0.5),int(output.shape[0]**0.5)))

        
        plt.subplot(2, 2, 1)
        plt.imshow(asm_image)
        plt.title("ASM")
        
        plt.subplot(2, 2, 2)
        plt.imshow(entropy_image)
        plt.title("Entropy")
        
        plt.subplot(2, 2, 3)
        plt.imshow(contrast_image)
        plt.title("Contrast")
        
        plt.subplot(2, 2, 4)
        plt.imshow(idm_image)
        plt.title("IDM")
        
        plt.suptitle("GLCM Features 315"+u"\N{DEGREE SIGN}")
        plt.savefig("GLCM Features 315 degrees")

        
        plt.show()
        
        # outputband90 = np.array(outputband90)
        
        # print()
        # outputband90 = outputband90 * 255/np.amax(outputband90)
        #         print(np.amax(outputband))
        # output_final.append([output.tolist(),output_ans1.tolist(),output_ans2.tolist()])
        for i in range(output.shape[0]):
            x = [output[i], output_ans1[i],output_ans2[i], output_idm[i]]
            output_final_315.append(x)
        # output_final.append(output)
        # output_final.append(output_ans1)
        # output_final.append(output_ans2)
        print(output[0],output_ans1[0],output_ans2[0], output_idm[0])
        # print(output_final)
        
        
        return np.array(output_final_315)
                
                

    def Input_filter_size(self,img_array,filter_size):
        
        self.filter_size = filter_size
        print("You have entered: "+str(self.filter_size))
        outputband315=self.convolution(img_array)
        
        print(outputband315.shape)
        
#        print("0 degree")

        
        # plt.subplot(2, 2, 2)
        # plt.imshow(outputband90)
        # plt.title("90 degree")
        

        return outputband315




""" Compute GLCM Features for KMeans """


# class glcm_features():
    
#     def __init__(self,img_array,kmeans):
#         self.img_array=img_array
#         self.kmeans=kmeans



global final_image
    
def outputs(img_array,filter_size,k):
    global final_image
    # print("glcm_features_outputs")
    print("GLCM_FeatureExtraction started at ", time.time())
    output_padded_0=GLCM_FeatureExtraction_0(img_array,filter_size)
    output_padded_45=GLCM_FeatureExtraction_45(img_array,filter_size)
    output_padded_90=GLCM_FeatureExtraction_90(img_array,filter_size)
    output_padded_135=GLCM_FeatureExtraction_135(img_array,filter_size)
    output_padded_180=GLCM_FeatureExtraction_180(img_array,filter_size)
    output_padded_270=GLCM_FeatureExtraction_270(img_array,filter_size)
    output_padded_225=GLCM_FeatureExtraction_225(img_array,filter_size)
    output_padded_315=GLCM_FeatureExtraction_315(img_array,filter_size)
    
    
    output_final_0 = (output_padded_0.Input_filter_size(img_array,filter_size))
    output_final_0 = output_final_0.reshape(output_final_0.shape[0],output_final_0.shape[1])
    # print(output_final_0.shape)
    output_final_45 = (output_padded_45.Input_filter_size(img_array,filter_size))
    output_final_45 = output_final_45.reshape(output_final_45.shape[0],output_final_45.shape[1])
    
    output_final_90 = (output_padded_90.Input_filter_size(img_array,filter_size))
    output_final_90 = output_final_90.reshape(output_final_90.shape[0],output_final_90.shape[1])
    
    output_final_135 = (output_padded_135.Input_filter_size(img_array,filter_size))
    output_final_135 = output_final_135.reshape(output_final_135.shape[0],output_final_135.shape[1])
    
    output_final_180 = (output_padded_180.Input_filter_size(img_array,filter_size))
    output_final_180 = output_final_180.reshape(output_final_180.shape[0],output_final_180.shape[1])
    
    output_final_270 = (output_padded_270.Input_filter_size(img_array,filter_size))
    output_final_270 = output_final_270.reshape(output_final_270.shape[0],output_final_270.shape[1])
    
    output_final_225 = (output_padded_225.Input_filter_size(img_array,filter_size))
    output_final_225 = output_final_225.reshape(output_final_225.shape[0],output_final_225.shape[1])
    
    output_final_315 = (output_padded_315.Input_filter_size(img_array,filter_size))
    output_final_315 = output_final_315.reshape(output_final_315.shape[0],output_final_315.shape[1])
    
    
    # print(output_final_0)               
    # print(output_final_45) 
    # print(output_final_90) 
    # print(output_final_135) 
    # print(output_final_180) 
    # print(output_final_315) 
    
    # print(output_final_0.shape)               
    # print(output_final_45.shape) 
    # print(output_final_90.shape) 
    # print(output_final_135.shape) 
    # print(output_final_180.shape) 
    print(output_final_270.shape)
    
    
    output_features = np.hstack((output_final_0,output_final_45,output_final_90,output_final_135,output_final_180,output_final_270,output_final_225,output_final_315))
  
    print("Kmeans has started at ", time.time())
    kmeans = KMeans(n_clusters=k, random_state=0).fit(output_features)
    kmeans_labels = (kmeans.labels_)
    kmeans_labels = kmeans_labels * 255 / np.max(kmeans_labels)
    final_image = kmeans_labels.reshape(int(kmeans_labels.shape[0]**0.5),int(kmeans_labels.shape[0]**0.5))
    # final_image.shape
    print("Kmeans has ended... Time Taken:" ,time.time())
    final_image = final_image.reshape(int(kmeans_labels.shape[0]**0.5), int(kmeans_labels.shape[0]**0.5))
    # final_image
    # return final_image
    plt.subplot(1,1,1)
    
    plt.imshow(final_image)
    plt.title("Texture Segments")
    
    # plt.suptitle("GLCM Features 0"+u"\N{DEGREE SIGN}")
    plt.savefig("Texture Segments")

    
    plt.show()
 
def just_for_returning():
     global final_image
     print(final_image)
     return final_image
        
    
    
    # plt.imshow("Output.png",final_image)
    # root = Tk()
    # root.title('Output')
    # # root.iconbitmap('logo.png')
    # root.configure(background='#077089')
    # display = cv2.imshow(final_image)
    # image_label = tk.Label(root, image=display)
    # image_label.grid(column=2, row=4,padx=10, pady=10)
    # root.mainloop()