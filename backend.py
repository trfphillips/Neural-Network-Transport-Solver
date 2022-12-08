import pandas as pd



#Function to import excel file
def import_excel(file_name):
    x = pd.read_excel(file_name+'.xlsx')
    x = x.iloc[:,:].values
    return x