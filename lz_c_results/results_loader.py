import pickle

pickle_in = open("uni_results.pickle","rb")
uni_results = pickle.load(pickle_in)

pickle_in = open("multi_results.pickle","rb")
multi_results = pickle.load(pickle_in)