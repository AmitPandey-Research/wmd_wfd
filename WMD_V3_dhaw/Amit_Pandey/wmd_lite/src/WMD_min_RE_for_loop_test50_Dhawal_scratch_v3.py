## To run as batch job


#imports:

# file imports

import numpy as np
import matplotlib.pyplot as plt
import gensim
import gensim.downloader
import os
from scipy.optimize import linprog
import nltk
from collections import defaultdict
from gensim.models import KeyedVectors
import sklearn
import scipy
import time
from multiprocessing import Pool
from nltk.corpus import stopwords
import json
from multiprocessing import Pool
import time
import pickle
from gensim import models

#nltk.download('stopwords')
#nltk.download('punkt')
# nltk.download('wordnet')

print("code started and imports done \n")




print("\n Directory check :\n")
print("\n /scratch/Amit_Pandey/wmd_lite/files:", os.listdir("/scratch/Amit_Pandey/wmd_lite/files"))
print(" \n os.listdir /scratch/Amit_Pandey/gensim-data/word2vec-google-news-300/:",
      os.listdir("/scratch/Amit_Pandey/gensim-data/word2vec-google-news-300/"))

print("\n printing current nltk path and adding to the path:")
print(nltk.data.path)

nltk.data.path.append("/scratch/Amit_Pandey/nltk_data")

print("\n",nltk.data.path) 




def sentence_preprocess(embed_dict, sentence,lowercase = 1, strip_punctuation = 1,  remove_stopwords = 1,removedigit = 1):
    ''' 1 : True, 0 : False : Lowercase, Strip puncutation, Remove Stopwords, removedigit'''

    stop_words = list(stopwords.words('english'))

    if lowercase == 1:
        sentence = sentence.lower()

    sentence_words = nltk.word_tokenize(sentence)

    if strip_punctuation == 1 and removedigit == 1:
        sentence_words = [word for word in sentence_words if word.isalpha()] 
        


    if remove_stopwords == 1:
        sentence_words = [word for word in sentence_words if not word in stop_words]
    
    ## to remove those words which are not in the embeddings that we have.
    
    sentence_words = [word for word in sentence_words if word in embed_dict.keys()]



    return sentence_words




embeddingtype = None
embd_model = None



## to load from embedding text files:
## have used this to load glove vectors and not word2vec

def load_glove(embeddingtype):
    
    if embeddingtype == 3:
        i = 300
        
        a_file = open("/scratch/Amit_Pandey/wmd_lite/files/reduced_glove_embedding_300.json", "r")
        output = json.load(a_file)
        print(len(output.keys()))
        a_file.close()
        
        embeddings_dict = output
        
    if embeddingtype == 4:
        i = 200
    if embeddingtype == 5:
        i = 100
    if embeddingtype == 6:
        i = 50
    
    
#     embeddings_dict = defaultdict(lambda:np.zeros(i)) 
#     # defaultdict to take care of OOV words.
    
#     with open(f"../files/glove.6B.{i}d.txt",'r') as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             vector = np.asarray(values[1:], "float32")
#             embeddings_dict[word] = vector
        
    return embeddings_dict



def embeddings_setup(newembeddingtype):
    
    
    global embeddingtype
    global embd_model
    
    
    '''to avoid loading all the embeddings in the memory.'''
    
    ''''## Note : we are finding the embd matrix two times, ie once for each sentence in
        ## the pair of sentences.
        ## so this happens that embedding type is changed when find_embmatrix is called
        ## by the first sentence.
        The above line doesnt matter now as we not calling find_embmatrix , instead we setting up.
    '''
        
        
        
    if ( embeddingtype != newembeddingtype):
        #print("embdtype  entered :", embeddingtype != newembtype,"\n")
        #print("embd_model type changed to :", type(embd_model),"\n" )
        
        embeddingtype = newembeddingtype
        
        #embd_model = embeddings_setup(embeddingtype) #adictionary
        
        #print("embd_model type changed to :", type(embd_model),"\n" )
        #to make sure that we don't download the embeddings again and again,
        # we will check if the embedding type is same as the old one
        # and update global embd_model, vrna next time vo use hi nhi ho payega.
    
    
    
    
    
    if embeddingtype == 1:
        
        ## To load from scratch:
        
        w = models.KeyedVectors.load_word2vec_format(
        '/scratch/Amit_Pandey/gensim-data/word2vec-google-news-300/GoogleNews-vectors-negative300.bin', binary=True)
        
        embedding = w
        
        #embedding = KeyedVectors.load('google300w2v.kv', mmap='r')
        ## This will be slower but will prevent kernel from crashing.
        
        ## comment the above line and uncomment this if you have sufficient RAM:
        
        #w2v_emb = gensim.downloader.load('word2vec-google-news-300')
        
    if embeddingtype == 2:
        print('Normalised word2vec not loaded, will get it soon')
        embedding = None
    
    if embeddingtype in (3,4,5,6):
        embedding = load_glove(embeddingtype)
        
    
    embd_model = embedding
    
    
        
def find_embdMatrix(sentence):
    global embeddingtype
    global embd_model
    #print(" global embedding type being passed is :", embeddingtype,"\n")
    #print("embedding type received by the find emb matrix is :", newembtype,"\n")
    #print("embd model type is :", type(embd_model),"\n")
    
    sent_mtx = []
    
    
    ##commented lines moved to embedding setup.
    
#     ''''## Note : we are finding the embd matrix two times, ie once for each sentence in
#     ## the pair of sentences.
#     ## so this happens that embedding type is changed when find_embmatrix is called
#     ## by the first sentence
#     '''
#     if ( embeddingtype != newembtype):
#         print("if embdtype part entered :", embeddingtype != newembtype,"\n")
        
#         embeddingtype = newembtype
#         embd_model = embeddings_setup(embeddingtype) #adictionary
        
#         print("embd_model type changed to :", type(embd_model),"\n" )
#     #to make sure that we don't download the embeddings again and again,
#     # we will check if the embedding type is same as the old one
#     # and update global embd_model, vrna next time vo use hi nhi ho payega.
    
    #print("embd_model type changed to :", type(embd_model),"\n" )
    for word in sentence:
        word_emb = embd_model[word]
        sent_mtx.append(word_emb)
    
    sent_mtx = np.array(sent_mtx).reshape(len(sentence),-1)

    return sent_mtx




def wasserstein_distance(pi, qj, D, cost = 'min'):
        """Find Wasserstein distance through linear programming
        p.shape=[m], q.shape=[n], D.shape=[m, n]
    
        suppose doc1 has m words and doc2 has n words, then an mxn array would be formed, 
        having distance of each word in doc1 to that of doc2.
    
    
    
        p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
        """
        A_eq = [] # a list which will later be converted to array after appending.
        for i in range(len(pi)): # len = number of words.
            A = np.zeros_like(D) # a 2d array made with the shape of D.  
            A[i, :] = 1 
            #print("Dshape, len pi till here :",D.shape,len(pi),"\n")
            
            # to make summation over "i" of Tij = pi, ie total / sum of outflow
            ## from one word is equal to its pi (normalized bag of word/ frequency/density)
            ## ex : if 2x3 D:
            ##T1,1 + T1,2 + T1,3 + 0 T2,1 + 0 T2,2 + 0 T2,3 = P1 and so on for every i,
            ## ie for each word in the doc1
            
            
            #print("A.shape", A.shape,"\n")
            A_eq.append(A.reshape(-1)) ## reshape(-1) flatens and then appending in A_eq.
            
            #print(A_eq,"Aeq\n")
            
            
            
            ## A_eq will be (m+n)x(m.n)
    
        for i in range(len(qj)):
            A = np.zeros_like(D)
            A[:, i] = 1 ## summation over "j" this time, so this time for different rows, 
            ## over a column "j" which refers to doc2, ie total incoming flow = qj density
            A_eq = list(A_eq)
            A_eq.append(A.reshape(-1))
            A_eq = np.array(A_eq)
        
        #print(A_eq.shape,A_eq)
       
        b_eq = np.concatenate([pi, qj])
        D = D.reshape(-1)
        #print("Dshape:",D.shape)
        if cost == 'max':
            D = D*(-1)
        
        result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1]) ## removing redundant to make 
        ## solution more robust.
        return np.absolute(result.fun), result.x , D.reshape((len(pi),len(qj)))  ## fun returns the final optimized value, x returns each value of xi,j that is the array

    
def relaxed_distance(pi,qj,D,cost='min'):
    
    # to find relaxed we just add the min/max cost directly using the least distance for pi to qj.
    
    # D is calculated from P to Q ie P in rows and Q in columns, To find Q to P we will transpose 
    if cost == 'min':
        p_to_q = np.dot(D.min(axis=1),pi)
        q_to_p = np.dot(D.T.min(axis=1),qj)
        
        return max(p_to_q,q_to_p)
    
    if cost == 'max':
        
        p_to_q = np.dot(D.max(axis=1),pi)
        q_to_p = np.dot(D.T.max(axis=1),qj)
        
        return min(p_to_q,q_to_p), None, D
        
        
    
    
class WMD:
    
    ''' wmd type = normal/relaxed, costtype = min/max.
    Enter Two sentence strings, cost = max if you want to try 
    max cost max flow version, embeddingtype = 1 for word2vec, 2 = normalized
    word2vec, 3 = glove300d, 4 = glove200d, 5 = glove100d 6 = glove50d'''
    
    def __init__(self,embeddingtype, wmd_type = 'normal', costtype='min'):
        
        
        self.cost = costtype
        
        self.embeddingtype = embeddingtype 
        self.wmd_type = wmd_type
        
        
        ## setting up the embeddings
        
        embeddings_setup(self.embeddingtype)
        
        
        
        
    #def word_count(self):
#         self.sent1_dic = defaultdict(int)
#         self.sent2_dic = defaultdict(int)
        
#         for word in sorted(sentence_preprocess(self.sent1)):
#             self.sent1_dic[word] += 1
            
#         for word in sorted(sentence_preprocess(self.sent2)):
#             self.sent2_dic[word] += 1
        
#         return dict(self.sent1_dic), dict(self.sent2_dic)



#     def wasserstein_distance(self, pi, qj, D):
#         """Find Wasserstein distance through linear programming
#         p.shape=[m], q.shape=[n], D.shape=[m, n]
    
#         suppose doc1 has m words and doc2 has n words, then an mxn array would be formed, 
#         having distance of each word in doc1 to that of doc2.
    
    
    
#         p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
#         """
#         A_eq = [] # a list which will later be converted to array after appending.
#         for i in range(len(pi)): # len = number of words.
#             A = np.zeros_like(D) # a 2d array made with the shape of D.  
#             A[i, :] = 1 
#             # to make summation over "i" of Tij = pi, ie total / sum of outflow
            ## from one word is equal to its pi (normalized bag of word/ frequency/density)
            ## ex : if 2x3 D:
            ##T1,1 + T1,2 + T1,3 + 0 T2,1 + 0 T2,2 + 0 T2,3 = P1 and so on for every i,
            ## ie for each word in the doc1
        
#             A_eq.append(A.reshape(-1)) ## reshape(-1) flatens and then appending in A_eq.
            ## A_eq will be (m+n)x(m.n)
    
#         for i in range(len(qj)):
#             A = np.zeros_like(D)
#             A[:, i] = 1 ## summation over "j" this time, so this time for different rows, 
#             ## over a column "j" which refers to doc2, ie total incoming flow = qj density
#             A_eq.append(A.reshape(-1))
#             A_eq = np.array(A_eq)
        
#         print(A_eq.shape,A_eq)
       
#         b_eq = np.concatenate([pi, qj])
#         D = D.reshape(-1)
#         if self.cost == 'max':
#             D = D*(-1)
        
#         result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1]) ## removing redundant to make 
#         ## solution more robust.
#         return result.fun, result.x  ## fun returns the final optimized value, x returns each value of xi,j that is the array

    
    def word_mover_distance(self,sentence1,sentence2):
        
        self.sent1 = sentence1
        #print(self.sent1 ,"\n")
        self.sent2 = sentence2
        #print(self.sent2 ,"\n")
        
        
        self.sent1_dic = defaultdict(int)
        self.sent2_dic = defaultdict(int)
        
        for word in sorted(sentence_preprocess(embd_model,self.sent1)): # sorted to have better
            self.sent1_dic[word] += 1 # idea of the sequence of the words. Creating BOW here
            
        for word in sorted(sentence_preprocess(embd_model,self.sent2)): #creating BOW from sorted sequence
            self.sent2_dic[word] += 1
        
        
        self.sent1_dic = dict(self.sent1_dic) # converted from default dict to dict.
        self.sent2_dic = dict(self.sent2_dic) # because following operations work on dict
        
        
        #print(self.sent1_dic ,"\n")
        #print(self.sent2_dic ,"\n")
        
        
        ## Now we will store a list/array of all the words in each sentence (in alphabetically sorted order)
        ## we will store corresponding count, and then corresponding Normalised count.
        self.sent1_words = np.array(list(self.sent1_dic.keys())) #dictionary keys converted to list than array
        self.sent1_counts = np.array(list(self.sent1_dic.values()))
        
        self.sent2_words = np.array(list(self.sent2_dic.keys()))
        self.sent2_counts = np.array(list(self.sent2_dic.values()))
        
        
        #print(self.sent1_words ,"\n")
        #print(self.sent1_counts ,"\n")
        
        #print(self.sent2_words ,"\n")
        #print(self.sent2_counts ,"\n")
        
        #dictionary values cant be converted into an array directly, hence the
        #list step.
        
        #print("embedding type being passed is :", self.embeddingtype,"\n")
        self.sent1_embmtx = find_embdMatrix(self.sent1_words)
        #print(self.sent1_embmtx.shape,"sent1emb\n")
        self.sent2_embmtx = find_embdMatrix(self.sent2_words)
        #print(self.sent2_embmtx.shape,"sent2emb\n")
        
        self.pi = self.sent1_counts/np.sum(self.sent1_counts) #NBOW step from BOW
        #print(self.pi,"self.pi\n")
        self.qj = self.sent2_counts/np.sum(self.sent2_counts)
        #print(self.qj,"self.qj\n")
        
        self.D = np.sqrt(np.square(self.sent1_embmtx[:, None] - self.sent2_embmtx[None, :]).sum(axis=2)) 
        #print(self.D.shape,"Dshape \n")
        ## programmers sought used mean instead of sum.
        ## scipy cdist can be used as well.
        
        if self.wmd_type == 'normal':
            return wasserstein_distance(self.pi, self.qj, self.D, self.cost)
        
        
        if self.wmd_type == 'relaxed':
            return relaxed_distance(self.pi,self.qj,self.D,self.cost)
        
print("\n FUNCTIONS DEFINITION OVER AND DATA LOADING STARTED\n")


## KNN

Train_BBCsport_sent = np.load("/scratch/Amit_Pandey/wmd_lite/files/Train_BBCsport_sent.npy")
Train_BBCsport_label = np.load("/scratch/Amit_Pandey/wmd_lite/files/Train_BBCsport_label.npy")
Test_BBCsport_sent = np.load("/scratch/Amit_Pandey/wmd_lite/files/Test_BBCsport_sent.npy")
Test_BBCsport_label = np.load("/scratch/Amit_Pandey/wmd_lite/files/Test_BBCsport_label.npy")


print("\n DATA LOADING ENDED\n")

#for i in range(5):
    #print(Test_BBCsport_label[i],"\n",Test_BBCsport_sent[i])
    

print("##################Train details:\n")

#for i in range(5):
    #print(Train_BBCsport_label[i],"\n",Train_BBCsport_sent[i])


# embeddingtype = 3
# model = WMD(embeddingtype,wmd_type = 'relaxed', costtype='max')

            
    
no_testdocs = len(Test_BBCsport_sent)
no_testlabels = len(Test_BBCsport_label)
#no_testdocs,no_testlabels



actual_category = []
predicted_category = []

    



#import time
st = time.time()
print("\n MODEL INITIALIZATION STARTED\n")
embeddingtype = 3
model = WMD(embeddingtype,wmd_type = 'normal', costtype='min')

print("\n MODEL INITIALIZATION OVER\n")

result_Dhawal = []
test_finished_Dhawal = []


def predict_Category(i):
    global result 
    global test_finished
    
    prediction_dictionary = {}
    sentence = Test_BBCsport_sent[i]
    
    distance_fromTrainset = []
    
    for j in range (len(Train_BBCsport_sent)):
    #for j in range (10): #number of train
        ## Find totalcost ie distance between sentence passed from test set to each sentence 
        ## in training set. and then append in the list.
        
        #print(sentence)
        #print(Train_BBCsport_sent[i])
        
        print(f"\nTrain{j}")
        
        Totalcost, Tcoeff, Distancematx = model.word_mover_distance(sentence,Train_BBCsport_sent[j])
        print(f" distance btwn test{i} and train{j} :", Totalcost,"\n")
        #print(Totalcost)
        distance_fromTrainset.append(Totalcost)
        
    distance_fromTrainset = np.array(distance_fromTrainset)
    #print('distance from train set array:',distance_fromTrainset)
    
    arr1indx = distance_fromTrainset.argsort()
    print("index of distance in increasing order is:", arr1indx, "\n")
    
    #print("Distance and label sorted from test set",distance_fromTrainset[arr1indx[::1]], "\n",Train_BBCsport_label[arr1indx[::1]],"\n","Sentences: \n",Train_BBCsport_sent[arr1indx[::1]]) 
    
    
    ## Taking for different values of K
    
    #k = 5
    sorted_distance_fromTrainset_k5 = distance_fromTrainset[arr1indx[::1]][:5]
    sorted_labels_k5 = Train_BBCsport_label[arr1indx[::1]][:5]
    
    predicted_cat_k5 = scipy.stats.mode(sorted_labels_k5)[0]
    #print("pred 5",predicted_cat_k5)
   

    #k = 7
    sorted_distance_fromTrainset_k7 = distance_fromTrainset[arr1indx[::1]][:7]
    sorted_labels_k7 = Train_BBCsport_label[arr1indx[::1]][:7]
    
    predicted_cat_k7 = scipy.stats.mode(sorted_labels_k7)[0]
    #print("pred 7",predicted_cat_k7)

    #k = 11
    sorted_distance_fromTrainset_k11 = distance_fromTrainset[arr1indx[::1]][:11]
    sorted_labels_k11 = Train_BBCsport_label[arr1indx[::1]][:11]
    
    predicted_cat_k11 = scipy.stats.mode(sorted_labels_k11)[0]
    #print("pred 11",predicted_cat_k11)

    #k = 15
    sorted_distance_fromTrainset_k15 = distance_fromTrainset[arr1indx[::1]][:15]
    sorted_labels_k15 = Train_BBCsport_label[arr1indx[::1]][:15]
    
    predicted_cat_k15 = scipy.stats.mode(sorted_labels_k15)[0]
    #print("pred 15",predicted_cat_k15)

    #k = 21
    sorted_distance_fromTrainset_k21 = distance_fromTrainset[arr1indx[::1]][:21]
    sorted_labels_k21 = Train_BBCsport_label[arr1indx[::1]][:21]
    
    predicted_cat_k21 = scipy.stats.mode(sorted_labels_k21)[0]
    #print("pred 21",predicted_cat_k21)


    #print(sorted_distance_fromTrainset,sorted_labels)
    prediction_dictionary[i] = [Test_BBCsport_label[i],
                                arr1indx[:30].tolist(),
                                Train_BBCsport_label[arr1indx[::1]][:30].tolist(),
                                distance_fromTrainset[arr1indx[::1]][:30].tolist(),
                                [predicted_cat_k5.tolist(),predicted_cat_k7.tolist(), predicted_cat_k11.tolist(),predicted_cat_k15.tolist(),
                                 predicted_cat_k21.tolist()]]
    
    
    result_Dhawal.append(prediction_dictionary)
    test_finished_Dhawal.append(i)
    
    
    with open('../results/result_Dhawal.pickle', 'wb') as handle:
        pickle.dump(result_Dhawal, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('../results/test_finished_Dhawal.pickle', 'wb') as handle:
        pickle.dump(test_finished_Dhawal, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    
    
    
    #return np.array([predicted_cat_k5,predicted_cat_k7,predicted_cat_k11,predicted_cat_k15,predicted_cat_k21])   
        
    
    
    
no_testdocs = len(Test_BBCsport_sent)
no_testlabels = len(Test_BBCsport_label)
#no_testdocs,no_testlabels



actual_categories = []
predicted_categories_list = []
for i in range (50,220): # number of test
    print(f" \n ################### Test{i} #############","\n")
    predict_Category(i)
    
    #actual_categories.append(Test_BBCsport_label[i]) 
    #pred_category = predict_Category(Test_BBCsport_sent[i])
    #print(pred_category)
    #predicted_categories_list.append(pred_category)
    

et = time.time() 

print("time taken:",et-st)


    



   
 

