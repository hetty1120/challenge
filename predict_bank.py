
# coding: utf-8

# In[1]:


from keras.models import load_model
import numpy as np
from keras.models import Model
import cv2
from PIL import Image
import argparse


# In[2]:


def change_image_size(temp):

    # if the size is big, crop it
    if temp.shape[0] > 1600 or temp.shape[1] > 1600:
        # if the height is larger than width, we can crop the image since the logo usually appears on the top of the page
        if temp.shape[0] > temp.shape[1]:
            temp = temp[:temp.shape[0]//4,:,:]
        
        ratio_height = 600/temp.shape[0]
        ratio_width = 1600/temp.shape[1]
        
        
        ratio = min(ratio_height,ratio_width)
        if ratio < 1:
            temp = cv2.resize(temp, (int(temp.shape[1]*ratio),int(temp.shape[0]*ratio)))
            
    return temp


# In[225]:


def model_predict(image):
    image = cv2.resize(image, (105,105))
    image = image.reshape((1,) + image.shape)
    result = model.predict([image])[0]
    
    return result


# In[226]:


def check_logo(temp):
    # if logo is not like a square, change the size with filling white space around it
    if temp.shape[0]/temp.shape[1] < 0.7 or temp.shape[1]/temp.shape[0] < 0.7:
        if temp.shape[0] > temp.shape[1]:
            add = np.full((temp.shape[0],temp.shape[0]-temp.shape[1],3),1,dtype=float)
            temp = np.concatenate((temp,add),axis=1)
        else:
            add = np.full((temp.shape[1]-temp.shape[0],temp.shape[1],3),1,dtype=float)
            temp = np.concatenate((temp,add))
    
    result = model_predict(temp)
    
    max_index = np.where(result==np.max(result))[0][0]
    if max_index != 4 and result[max_index] > 0.7:
        return class_dic[max_index]
    return 'Other'


# In[228]:


def check_full_image(image):
    stride = 25
    sizes = [(100,100),(100,150),(200,200),(200,300)]
    max_predict = 0
    ans_index = -1
    
    for i in range(0,image.shape[0]-99,stride):
        for j in range(0,image.shape[1]-99,stride):
            for k in range(len(sizes)):
                i_bottom = i + sizes[k][0]
                j_right = j + sizes[k][1]
                if i_bottom <= image.shape[0] and j_right <= image.shape[1]:
                    temp = image[i:i_bottom,j:j_right,:]
                    # if the cropped image is white, continue
                    if ((sizes[k][0]*sizes[k][1]*3 - np.sum(temp)) < 1) or (np.sum(temp) - 0 < 1):
                        continue
                    add = np.full((sizes[k][1]-sizes[k][0],sizes[k][1],3),1,dtype=float)
                    temp = np.concatenate((temp,add))
                    result = model_predict(temp)
                    max_index = np.where(result==np.max(result))[0][0]
                    if max_index != 4 and result[max_index] > max_predict:
                        max_predict = result[max_index]
                        ans_index = max_index
                        if max_predict >= 0.95:
                            return class_dic[ans_index]
    
    return 'Other'


# In[230]:


parser = argparse.ArgumentParser(description='bank_prediction')
parser.add_argument('string',type=str,help='file name')
args = parser.parse_args()
file_path = args.string

model = load_model('logo_model_3.hdf5')
class_dic = {0:'Bank Of America', 1:'Capital One', 2:'JPMorgan Chase', 
             3:'Citigroup Inc', 4:'Other', 5:'Wells Fargo'}

image = Image.open(file_path).convert('RGB')
image = np.array(image)[:,:,:3]/255
result = check_logo(image)
if result == 'Other' and (image.shape[0]>1000 or image.shape[1]>1000):
    result = check_full_image(image)

print('Document belongs to organization: {}'.format(result))

