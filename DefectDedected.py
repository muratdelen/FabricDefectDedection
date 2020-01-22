
import os
from shutil import copyfile
import cv2

from numpy import save, load
from random import seed, random

from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
##---------------------------------------------------------------------------------------------------------------------------------------------
# FOLDER INFORMATION
##---------------------------------------------------------------------------------------------------------------------------------------------
dataSetFolder = "dataset/"
dataSetOrginalFolder = dataSetFolder + "original/"
dataSetTrain = dataSetFolder + "train/"
dataSetTest = dataSetFolder + "test/"
dataSetNoDefectFolder = dataSetOrginalFolder + "NODefect_images/"
dataSetDefectFolder = dataSetOrginalFolder + "Defect_images/"
dataSetTestRatio = 0.25
print(os.listdir(dataSetFolder))
#image resize
imageHeight = 224 #25
imageWidth = 224 #400
# dateset categories
Categories = ["Defect","NoDefect"]
#epochs times
Epochs = 100
#tensorflow CNN model
model = Sequential()

##---------------------------------------------------------------------------------------------------------------------------------------------
# IMAGES INFORMATION AND PREPARE DATASET FROM ONE FOLDER OR TWO FOLDER
##---------------------------------------------------------------------------------------------------------------------------------------------
"""
showImages function show image to analyzed.
can be change image size
"""
#showImages(dataSetNoDefectFolder,100,20)
#showImages(dataSetNoDefectFolder)
#showImages(dataSetDefectFolder)
def showImages(image_folder, img_width=None, img_height=None):
	i = 0
	pyplot.figure(image_folder)
	for image_full_name in os.listdir(image_folder):
		filename = image_folder + image_full_name
		image_data = cv2.imread(filename)
		if img_width != None and img_height != None:
			image_data = cv2.resize(image_data,(img_width, img_height))
		pyplot.subplot(330 + 1 + i)
		pyplot.title(image_full_name)
		pyplot.imshow(image_data,)
		i += 1
		if(i == 9):break
	# show the figure
	pyplot.show()

"""
prepareDataSetAsData function set train and test dataset as data from defined dataset images folder.
data save as data_photos.npy and data_labels.npy 
"""
#prepareDataSet(dataSetNoDefectFolder,dataSetDefectFolder,200,200)
def prepareDataSetAsData(img_folder, first_object_name, second_object_name, img_width=None, img_height=None):
	print("Start to Prepare Dataset")
	photos, labels = list(), list()
	for image_full_name in os.listdir(img_folder):
		filename = img_folder + image_full_name		
		if image_full_name.startswith(first_object_name):
			img_label = 0.0
		elif image_full_name.startswith(second_object_name):
			img_label = 1.0
		else:
			img_label = 0.0
		image_data = cv2.imread(filename)
		if img_width != None and img_height != None:
			image_data = cv2.resize(image_data,(img_width, img_height))
		photos.append(image_data)
		labels.append(img_label)
	save(dataSetFolder + "data_photos.npy",photos)
	save(dataSetFolder + "data_labels.npy",labels)
	print(photos.shape, labels.shape)

"""
prepareDataSetAsData2 function set train and test dataset as data from defined 2 dataset images folder.
data save as data_photos.npy and data_labels.npy 
"""
#prepareDataSet2(dataSetNoDefectFolder,dataSetDefectFolder,200,200)
def prepareDataSetAsData2(first_object_folder, second_object_folder, img_width=None, img_height=None):
	print("Start to Prepare Dataset")
	photos, labels = list(), list()
	for image_full_name in os.listdir(first_object_folder):
		filename = first_object_folder + image_full_name		
		img_label = 0.0
		image_data = cv2.imread(filename)
		if img_width != None and img_height != None:
			image_data = cv2.resize(image_data,(img_width, img_height))
		photos.append(image_data)
		labels.append(img_label)
	for image_full_name in os.listdir(second_object_folder):
		filename = second_object_folder + image_full_name		
		img_label = 1.0
		image_data = cv2.imread(filename)
		if img_width != None and img_height != None:
			image_data = cv2.resize(image_data,(img_width, img_height))
		photos.append(image_data)
		labels.append(img_label)
	save(dataSetFolder + "data_photos.npy",photos)
	save(dataSetFolder + "data_labels.npy",labels)
	print(photos.shape, labels.shape)

"""
prepareDataSetAsImage function set train and test dataset as image from defined dataset images folder.
data save as data_photos.npy and data_labels.npy 
"""
#prepareDataSetAsImage(dataSetOrginalFolder,"cat","dog")
def prepareDataSetAsImage(img_folder, first_object_name, second_object_name):
	print("Start to Prepare Dataset")
	# create directories
	subdirs = ['train/', 'test/']
	for subdir in subdirs:
		# create label subdirectories
		labeldirs = [first_object_name + "/", second_object_name + "/"]
		for labldir in labeldirs:
			newdir = dataSetFolder + subdir + labldir
			os.makedirs(newdir, exist_ok=True)
	# seed random number generator
	seed(1)
	# copy training dataset images into subdirectories
	for image_full_name in os.listdir(img_folder):
		filename = img_folder + image_full_name  
		save_directory = dataSetFolder + "train/"
		if random() < dataSetTestRatio :
			save_directory = dataSetFolder + "test/"
		if image_full_name.startswith(first_object_name):
			save_directory += first_object_name + "/" + image_full_name
		elif image_full_name.startswith(second_object_name):
			save_directory += second_object_name + "/" + image_full_name
		copyfile(filename,save_directory)
	print("Finish Dataset")

"""
prepareDataSetAsImage function set train and test dataset as image from defined dataset images folder.
data save as data_photos.npy and data_labels.npy 
"""
#prepareDataSetAsImage2("cat",dataSetCatFolder,"dog",dataSetDogFolder)
def prepareDataSetAsImage2(first_object_name, first_object_folder, second_object_name, second_object_folder):
	print("Start to Prepare Dataset")
	# create directories
	subdirs = ['train/', 'test/']
	for subdir in subdirs:
		# create label subdirectories
		labeldirs = [first_object_name + "/", second_object_name + "/"]
		for labldir in labeldirs:
			newdir = dataSetFolder + subdir + labldir
			os.makedirs(newdir, exist_ok=True)
	# seed random number generator
	seed(1)
	# copy training dataset images into subdirectories
	for image_full_name in os.listdir(first_object_folder):
		filename = first_object_folder + image_full_name  
		save_directory = dataSetFolder + "train/"	  
		if random() < dataSetTestRatio :
			save_directory = dataSetFolder + "test/"
		save_directory += first_object_name + "/" + image_full_name
		copyfile(filename,save_directory)
	for image_full_name in os.listdir(second_object_folder):
		filename = second_object_folder + image_full_name  
		save_directory = dataSetFolder + "train/"	  
		if random() < dataSetTestRatio :
			save_directory = dataSetFolder + "test/"
		save_directory += second_object_name + "/" + image_full_name
		copyfile(filename,save_directory)
	print("Finish Dataset")

"""
loadDatasetWithdata function load dataset from data
"""
def loadDatasetWithdata():
	photos = load(dataSetFolder + "data_photos.npy")
	labels = load(dataSetFolder + "data_labels.npy")
	print(photos.shape,labels.shape)
	return photos,labels
##---------------------------------------------------------------------------------------------------------------------------------------------
#  DEFINE MODEL AND START TRAIN THE MODEL
##---------------------------------------------------------------------------------------------------------------------------------------------
"""
defineModel function defined model as CNN
can be create model use tensorflow backend for special problems
"""
def defineModel():
	## Feature Learning
	#1.  block
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(imageHeight, imageWidth, 3)))
	model.add(MaxPooling2D((2, 2)))
	#2.  block
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	#3.  block
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	#4.  block
	model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	#5.  block
	model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	#6.  block
	model.add(Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))

	#model.add(Dropout(0.2))
	## Classification
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))	
	model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))	
	model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))	 
	#model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
	#model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
	#model.add(Dense(4, activation='relu', kernel_initializer='he_uniform'))
	#model.add(Dense(2, activation='relu', kernel_initializer='he_uniform'))
	#model.add(Dropout(0.5))

	model.add(Dense(1, activation='sigmoid'))
	#opt = SGD(lr=0.01, momentum=0.99)
	opt = SGD(lr=0.0001, momentum=0.99)
	#opt = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
	#model.add(Dense(1, activation=K.exp))
	#model.add(Dense(1, activation='relu', kernel_initializer='he_uniform'))
	#opt=Adam(lr=0.01)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	model.summary()
	return model

"""
Show train and test information
"""
def summarizeDiagnostics(info):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(info.history['loss'], color='blue', label='train')
	pyplot.plot(info.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(info.history['accuracy'], color='blue', label='train')
	pyplot.plot(info.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	pyplot.savefig('train_info_plot.png')
	pyplot.close()

"""
after prepare model and dataset, started train.
"""
def runTrainTest():
	# get model
	model = defineModel()
	# create data generator
	train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
	# prepare iterators
	train_it = train_datagen.flow_from_directory(dataSetTrain, class_mode='binary', batch_size=64, target_size=(imageHeight, imageWidth))
	test_it = test_datagen.flow_from_directory(dataSetTest, class_mode='binary', batch_size=64, target_size=(imageHeight, imageWidth))
	# fit model
	print("Start Train")
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=Epochs, verbose=1)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('Result>> %.3f' % (acc * 100.0))
	# serialize weights to HDF5
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	model.save("model.h5")
	model.save_weights("model_weights.h5")
	print("Saved model to disk")
	#model.save_widths('models/augmented_30_epochs.h5')
	# learning curves
	summarizeDiagnostics(history)

#showImages(dataSetNoDefectFolder,100,20)
#showImages(dataSetNoDefectFolder)
#showImages(dataSetDefectFolder)
#prepareDataSetAsImage2("NoDefect",dataSetNoDefectFolder,"Defect",dataSetDefectFolder)
runTrainTest()

##---------------------------------------------------------------------------------------------------------------------------------------------
# LOAD MODEL AND TEST DATASET IMAGES
##---------------------------------------------------------------------------------------------------------------------------------------------
def load_image(img_path, show=False):
	# (height, width, channels)
    img = image.load_img(img_path, target_size=(imageHeight, imageWidth, 3))
	 # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width)   
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)                                                                                                                                                                                                        # channels)
    img_tensor /= 255. 
    if show:
        pyplot.imshow(img_tensor[0])                           
        pyplot.axis('off')
        pyplot.show()
    return img_tensor
#load model from saved folder
model = load_model("model.h5")
image_folder = "dataset/test/Defect/"
defect_count = 0
nodefect_count = 0
#load test dataset images and show result
for image_full_name in os.listdir(image_folder):
	filename = image_folder + image_full_name
	new_image = load_image(filename)
	pred = model.predict(new_image)
	if pred > 0.5:
		nodefect_count += 1
		pred_int = 1
	else:
		defect_count += 1
		pred_int = 0
	result = Categories[pred_int]
	print(image_full_name,pred,pred_int,result)

print("Defect: " + str(defect_count))
print("NoDefect: " + str(nodefect_count))