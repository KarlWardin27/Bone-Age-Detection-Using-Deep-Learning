import cv2
import numpy as np
import shutil
import imutils
import pandas as pd 

def main():
    data = pd.read_csv("D:/Code_Me/Python/Bone_Age_Detection/TrainData/datasets_10832_15122_boneage-training-dataset.csv")
    Path = 'D:/Code_Me/Python/Bone_Age_Detection/TrainData/PNGFormat/'
    ResultPath = 'D:/Code_Me/Python/Bone_Age_Detection/Augmented data/Train Images/'
    counter = 1
    csvcount = 0
    # 15611

    #For CSV File
    df = pd.DataFrame(
        {
            "id":[ ],
            "boneage":[ ],
            "male":[ ]
        }
    )
    convert_dict = {'male': bool, 
                } 
    df = df.astype(convert_dict)

    for x in range(1377, 1478):
        Path1 = Path + str(x) + '.png'
        ResultPath1 = ResultPath + str(counter) + '.png'
        try:
            shutil.copyfile(Path1, ResultPath1)  
            image = cv2.imread(Path1)
            df.at[counter-1, 'id'] = counter
            df.at[counter-1, 'boneage'] = data['boneage'][csvcount]
            df.at[counter-1, 'male'] = data['male'][csvcount]
             
            for y in range(-90,91,10):
                
                if (y == 0):
                    continue
                    
                else:
                    counter = counter + 1
                    df.at[counter-1, 'id'] = counter
                    df.at[counter-1, 'boneage'] = data['boneage'][csvcount]
                    df.at[counter-1, 'male'] = data['male'][csvcount]
                    rotated = imutils.rotate_bound(image, y)
                    name = ResultPath +str(counter)+ ".png"
                    cv2.imwrite(name, rotated)
                                  
        except:
            continue

        counter = counter + 1     
        csvcount = csvcount + 1
        print(str(x) + " of 15611 " + str(counter))

    df.to_csv("D:/Code_Me/Python/Bone_Age_Detection/Augmented data/AugCSV.csv")

if __name__ == "__main__":
    main()