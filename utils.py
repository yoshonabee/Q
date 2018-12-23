import pickle
from argparse import ArgumentParser
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

def get_args():
    parser = ArgumentParser()
    parser.add_argument('model_path', help="modelpath")
    parser.add_argument('height', help='height')
    parser.add_argument('weight', help='weight')
    parser.add_argument('agent_num', help='agent number')
    parser.add_argument('-c', dest='cuda', help='cuda number', default='default')
    parser.add_argument("-k", help="keep train", dest="keep_train", default="default")

    return parser.parse_args()