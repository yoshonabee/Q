import pickle
#--------------------------------------------------------------------------------
#the class and function will use in the smart-Wall-Es' project
#--------------------------------------------------------------------------------
def save_object(fname, obj):
    #the function is used to save some data in class or object in .pkl file
    with open(fname, 'wb') as out_file:
        pickle.dump(obj, out_file)
    out_file.close()

def load_object(fname):
    #the function is used to read the data in .pkl file
    with open(fname, 'rb') as in_file:
        return pickle.load(in_file)
