import shutil
import pandas as pd 
import os


def main():

    #Example CSV File Path Input :- D:/Bone_Age_Detection/TrainData/datasets_10832_15122_boneage-training-dataset.csv

    DATA_TRAIN = pd.read_csv(input('Enter the Directory of CSV File For Train (Please make sure that your directories are separated by "/" instead of "\") '))
    DATA_TEST = pd.read_csv(input('Enter the Directory of CSV File For Test '))

    #Example Image Path Input :- D:/Code_Me/Python/Bone_Age_Detection/TrainData
    
    ORIGINAL_IMAGE_PATH_TRAIN = input('Enter the Directory of Training Images (Please make sure that there is no "/" at the end of the path) ')
    ORIGINAL_IMAGE_PATH_TEST = input('Enter the Directory of Testing Images (Please make sure that there is no "/" at the end of the path) ')

    #Creating the Directory for Storing the Bifurcated Images
    os.mkdir('C:/Bifurcated Data')
    os.mkdir('C:/Bifurcated Data/Test')
    os.mkdir('C:/Bifurcated Data/Train')
    os.mkdir('C:/Bifurcated Data/Test/Male')
    os.mkdir('C:/Bifurcated Data/Train/Male')
    os.mkdir('C:/Bifurcated Data/Test/Female')
    os.mkdir('C:/Bifurcated Data/Train/Female')

    #Resultant Path for Storing Images
    RESULT_IMAGE_PATH_TRAIN = 'C:/Bifurcated Data/Train/'
    RESULT_IMAGE_PATH_TEST = 'C:/Bifurcated Data/Test/'

    #Function written for Bifurcation
    Bifurcate(1377, 15611, DATA_TRAIN, ORIGINAL_IMAGE_PATH_TRAIN, RESULT_IMAGE_PATH_TRAIN)
    Bifurcate(4360, 4560, DATA_TEST, ORIGINAL_IMAGE_PATH_TEST, RESULT_IMAGE_PATH_TEST)

    print('Your File has been Saved to C:/Bifurcated Data')
   

#Function that Bifurcates Images named K.png all the way upto M.png, into Male and Female X-rays
def Bifurcate (k, m, DATA, Original_Image_Path, Result_Image_Path):
    csvcount = 0
    for x in range(k, m):
    
        #A Try-Catch Block used to make the algorithm work even if there is a missing image name. 
        try:
            Original_Path = Original_Image_Path + '/' + str(x) +'.png'

            if(DATA['male'][csvcount] == True):
                Result_Path = Result_Image_Path +'Male/'+ str(x) + '.png'
                shutil.copyfile(Original_Path, Result_Path)

            else:
                Result_Path = Result_Image_Path +'Female/'+ str(x) + '.png'
                shutil.copyfile(Original_Path, Result_Path)          
                                  
        except:
            continue

        csvcount = csvcount + 1
        Result_Path = ''


if __name__ == "__main__":
    main()