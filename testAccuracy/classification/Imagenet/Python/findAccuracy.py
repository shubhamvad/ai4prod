import pandas as pd

# File Gt from imagenet
pd_GT=pd.read_csv("validation-imagenet.csv",header=None,delimiter=",")
#File obtained from output model
pd_Detection=pd.read_csv("classification-Detection.csv")


counter5Prediction=0

counter1Prediction=0

totalImageProcessed=0

for index, row in pd_Detection.iterrows():
    #print("Indici detection ",row[0], row[1])
    
    totalImageProcessed=totalImageProcessed+1
    arraY_Detection_class=[]

    arraY_Detection_class.append(row[1])
    arraY_Detection_class.append(row[2])
    arraY_Detection_class.append(row[3])
    arraY_Detection_class.append(row[4])
    arraY_Detection_class.append(row[5])

    result=pd_GT.isin([row[0]])
    
    seriesObj = result.any()

    
    columnNames=list(seriesObj[seriesObj == True].index)

    

    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        selcetRow=None
        for row in rows:
            
            selcetRow=row
    # indice riga corrisponde al valore della riga corrispondente nel dataframe ground_truth 
    # con id immagine uguale a quellio dell'immagine processata dal ciclo for
    
    #print("indice riga", selcetRow)

    Gt_class= pd_GT.at[int(selcetRow),1]

    #print("gt_class " ,Gt_class)    

    #print("array_detection", arraY_Detection_class)

    
    if(int(Gt_class)==arraY_Detection_class[0]):
        counter1Prediction=counter1Prediction+1


    if(Gt_class in arraY_Detection_class):
        counter5Prediction=counter5Prediction + 1

    
#50000 numero di immagini nel validation set
print("Accuracy 1 prediction ", counter1Prediction/totalImageProcessed)

print("Accuracy 5 prediction ", counter5Prediction/totalImageProcessed)