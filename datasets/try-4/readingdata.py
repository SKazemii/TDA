import os, sys
import glob
import datetime
import pandas as pd
import numpy as np
import re
import shutil
import random

# get the input and output path
input_path = "casis/Dataset/cumulative_align_barefoot"
output_path = "casis/Dataset/categorical_imgs"

Metadata = np.load("casis/Dataset/PerFootMetaDataBarefoot.npy")


images = (glob.glob(os.path.join(input_path, '*.jpg')))#, key=os.path.getmtime)

ID=(Metadata[:,0]).astype(int)




data_index = list()
for i in range(len(images)):
    image_name = images[i].split("/")
    image_name = image_name[len(image_name) - 1]
    ind, _ = re.split(r"\.", image_name)
    _, ind = re.split(r"\_", ind)
    data_index.append(ind)





data_index = [int(i) for i in data_index]
sorted = Metadata[data_index,0]



df = pd.DataFrame({"path" : images, "ID" : sorted, "ind" : data_index})
print (df.head(22))


# saving the dataframe
df.to_csv("casis/Dataset/df.csv", index=True)
# sys.exit()




# footprints class names
class_names = df["ID"].unique()
print("[INFO] Subject ID = {}".format(class_names))




# get the class label limit
class_limit = class_names.__len__()
print("[INFO] Number of subjects = {}".format(class_limit))




# change the current working directory
if os.path.exists("casis/Dataset/categorical_imgs"):
    print("[INFO] The categorical_imgs folder exists")
else:
    # create a folder for each subjects
    os.system("mkdir casis/Dataset/categorical_imgs")

os.chdir(output_path)
# sys.exit()
# loop over the class labels
for x in range(class_limit):
    if not os.path.exists(str(class_names[x])):
        # create a folder for each subjects
        os.system("mkdir " + str(class_names[x]))



# loop over the images in the imgs folder
for ind in df.index:
    original_path = df["path"][ind]
    
    image = original_path.split("/")
    image = image[len(image) - 1]
    
    cur_path = ("./" + str(df["ID"][ind]) + str(df["ind"][ind]) + ".jpg")

    # os.system("cp " + "/Users/saeedkazemi/Desktop/temp/OSDN-1/" + original_path + " " + cur_path)
    os.system("cp " + "/Users/saeedkazemi/Desktop/temp/OSDN-1/" + original_path + " ../data/"+ str(int(df["ID"][ind])) +"-"+ str(df["ind"][ind]) + ".jpg")
   
    
# Organize data into train, valid, test dirs
os.chdir('../data')
if os.path.isdir('train/subject') is False:
    os.makedirs('train/subject')
    os.makedirs('train/others')
    os.makedirs('test/imposter')
    os.makedirs('test/subject')
    os.makedirs('test/others')

for i in random.sample(glob.glob('4-*'), 26):
    shutil.copy(i, 'train')
    shutil.move(i, 'train/subject')      
for i in random.sample(glob.glob('[1-5][0-9]-*'), 50):
    shutil.copy(i, 'train')
    shutil.move(i, 'train/others')
for i in random.sample(glob.glob('[6-9][0-9]-*'), 10):
    shutil.move(i, 'test/imposter')
for i in random.sample(glob.glob('[1-5][0-9]-*'), 10):
    shutil.copy(i, 'test')      
    shutil.move(i, 'test/others')      
for i in random.sample(glob.glob('4-*'), 4):
    shutil.copy(i, 'test')
    shutil.move(i, 'test/subject')


images = (glob.glob('/Users/saeedkazemi/Desktop/temp/OSDN-1/casis/Dataset/data/train/*.jpg'))#, key=os.path.getmtime)


data_cls = list()
for i in range(len(images)):
    image_name = images[i].split("/")
    image_name = image_name[len(image_name) - 1]
    print(re.split(r"\-", image_name))
    cls, _ = re.split(r"\-", image_name)
    data_cls.append(cls)





df = pd.DataFrame({"path" : images, "label" : data_cls})
print (df.head(22))


df.to_csv("/Users/saeedkazemi/Desktop/temp/OSDN-1/casis/Dataset/data/train/df.csv", index=True)



os.chdir('../../')
