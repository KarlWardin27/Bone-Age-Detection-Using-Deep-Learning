import numpy as np 
import pandas as pd 
import tensorflow as tf
import datetime, os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#library required for image preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from  tensorflow.keras.applications.xception import preprocess_input 

from tensorflow.keras.metrics import mean_absolute_error

from tensorflow.keras.layers import GlobalMaxPooling2D, Dense,Flatten
#from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop

def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age)) 


print(tf.__version__)


#loading dataframes
train_df = pd.read_csv('E:/Bone Age Detection/boneage-training-dataset.csv')
test_df = pd.read_csv('E:/Bone Age Detection/boneage-test-dataset.csv')

#appending file extension to id column for both training and testing dataframes
train_df['id'] = train_df['id'].apply(lambda x: str(x)+'.png')
#test_df['Case ID'] = test_df['Case ID'].apply(lambda x: str(x)+'.png') 


#finding out the number of male and female children in the dataset
#creating a new column called gender to keep the gender of the child as a string
train_df['gender'] = train_df['male'].apply(lambda x: 'male' if x else 'female')
print(train_df['gender'].value_counts())
print('Click X to proceed further')
sns.countplot(x = train_df['gender'])
plt.show()



#oldest child in the dataset
print('MAX age: ' + str(train_df['boneage'].max()) + ' months')

#youngest child in the dataset
print('MIN age: ' + str(train_df['boneage'].min()) + ' months')

#mean age is
mean_bone_age = train_df['boneage'].mean()
print('mean: ' + str(mean_bone_age))

#median bone age
print('median: ' +str(train_df['boneage'].median()))

#standard deviation of boneage
std_bone_age = train_df['boneage'].std()

#models perform better when features are normalised to have zero mean and unity standard deviation
#using z score for the training
train_df['bone_age_z'] = (train_df['boneage'] - mean_bone_age)/(std_bone_age)


print('Printing Dataset')
print(train_df.head())

print('-----------------------------------------------------')
print('----------------Preprocessing Steps------------------')
print('-----------------------------------------------------')

#splitting train dataframe into traininng and validation dataframes
df_train, df_valid = train_test_split(train_df, test_size = 0.2, random_state = 0)



#reducing down the size of the image 
img_size = 256
#datagen=ImageDataGenerator(preprocessing_function = preprocess_input,validation_split=0.25)


train_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
val_generator = ImageDataGenerator(preprocessing_function = preprocess_input)


#train data generator
train_generator = train_generator.flow_from_dataframe(
    dataframe = df_train,
    directory = 'E:/Bone Age Detection/boneage-training-dataset/boneage-training-dataset',
    x_col= 'id',
    y_col= 'bone_age_z',
    batch_size = 32,
    shuffle = True,
    seed=42,
    class_mode= 'raw',
    color_mode = 'rgb',
    target_size = (img_size, img_size))

#validation data generator
val_generator = val_generator.flow_from_dataframe(
    dataframe = df_valid,
    directory = 'E:/Bone Age Detection/boneage-training-dataset/boneage-training-dataset',
    x_col = 'id',
    y_col = 'bone_age_z',
    batch_size =32,
    shuffle = False,
    seed=42,
    class_mode = 'raw',
    color_mode = 'rgb',
    target_size = (img_size, img_size))

#test data generator
'''
test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

test_generator = test_data_generator.flow_from_directory(
    directory = '/content/drive/MyDrive/BE_Project/Bone_Age_Detection/boneage-test-dataset',
    shuffle = True,
    class_mode = None,
    color_mode = 'rgb',
    target_size = (img_size,img_size))
'''


print('-----------------------------------------------------')
print('----------------Generators Created-------------------')
print('-----------------------------------------------------')



model_1 = tf.keras.applications.xception.Xception(input_shape = (img_size, img_size, 3),
                                           include_top = False,
                                           weights = 'imagenet')
model_1.trainable = True
model_2 = Sequential()
model_2.add(model_1)
model_2.add(GlobalMaxPooling2D())
model_2.add(Flatten())
model_2.add(Dense(10, activation = 'relu'))
model_2.add(Dense(1, activation = 'linear'))


#compile model
model_2.compile(loss ='mse', optimizer= 'adam', metrics = [mae_in_months] )

print('-----------------------------------------------------')
print('-------------------Model Created---------------------')
print('-----------------------------------------------------')

#model summary
model_2.summary()

print('-----------------------------------------------------')
print('---------------------Training------------------------')
print('-----------------------------------------------------')

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

#print(STEP_SIZE_TRAIN, STEP_SIZE_VALID)

history = model_2.fit(train_generator,
                      steps_per_epoch = STEP_SIZE_TRAIN,
                      validation_data = val_generator,
                      validation_steps = 1,
                      epochs = 2,
                      verbose=1)


print('-----------------------------------------------------')
print('------------------Done Training----------------------')
print('-----------------------------------------------------')


acc = history.history['mae_in_months']
val_acc = history.history['val_mae_in_months']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(mae_in_months))

plt.plot(epochs, acc, 'r', label='Training Mean Average Error')
plt.plot(epochs, val_acc, 'b', label='Validation Mean Average Error')
plt.title('Training and validation Mean Average Error')
plt.legend(loc=0)
plt.figure()
plt.show()





