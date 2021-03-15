import numpy as np
import pandas as pd
import re

dataset = pd.read_csv("./dataset_NB", sep="\n", header = None)

data = pd.DataFrame(data = np.asarray(dataset), columns = ['text'])
data.insert(1, "class", 'NULL')

#removing the specified stopwords
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def removeStopWords(sentence):
    
    updated = list()
    for word in sentence:
        if word in stopwords:
            continue
        else:
            updated.append(word)
    return updated
    
#Preprocesses the dataset, removes punctuations and extra whitespaces, adds classes
def prepareDataset(df):
    
    for i in range(df.shape[0]):
        df['class'][i] = int(df['text'][i][-1])
        df['text'][i] = df['text'][i][:-1].lower()
        df['text'][i] = re.sub("[^\w\s]", " ", df['text'][i]).strip()
        df['text'][i] = df['text'][i].split()
        df['text'][i] = removeStopWords(df['text'][i])
        
    return df

final_data = prepareDataset(data)

#creating 7 fold dataset
shuffle_data = final_data.sample(frac = 1)

data1 = shuffle_data.iloc[:int(1000/7),:]
data2 = shuffle_data.iloc[int(1000/7):int(2000/7),:]
data3 = shuffle_data.iloc[int(2000/7):int(3000/7),:]
data4 = shuffle_data.iloc[int(3000/7):int(4000/7),:]
data5 = shuffle_data.iloc[int(4000/7):int(5000/7),:]
data6 = shuffle_data.iloc[int(5000/7):int(6000/7),:]
data7 = shuffle_data.iloc[int(6000/7):int(7000/7),:]

sevenfold = [data1, data2, data3, data4, data5, data6,data7]

accuracy = np.zeros([7,1])

#iterating 7 times, ignoring one fold in training each time
for i in range(7):

    #creating the training dataset
    temp = []
    for j in range(7):
        if(i == j):
            continue
        else:
            temp.append(sevenfold[j])
    
    train = pd.concat(temp)
    train = train.reset_index()
   
   #creating a vocabulary with every word in the entire dataset
    vocabulary = set()

    def vocabularySize():
        
        for i in range(final_data.shape[0]):
            for word in final_data['text'][i]:
                vocabulary.add(word)
                
        return vocabulary, len(vocabulary)

    vocab, size  = vocabularySize()

    #storing the frequency of each word in a positive and neagtive samples
    def returnTypeCount(df, vocab):
        
        #initialising from 1 to avoid the 0 probability misclassification
        positive_dict = dict.fromkeys(vocab, 1)
        negative_dict = dict.fromkeys(vocab, 1)
        negative_dict
        positive_count = 0
        negative_count = 0
        
        for i in range(df.shape[0]):
            if df['class'][i] == 1:
                for word in df['text'][i]:
                    positive_dict[word] += 1
                    positive_count+=1
            else:
                for word in df['text'][i]:
                    negative_dict[word] += 1
                    negative_count+=1
                    
        return positive_count, positive_dict, negative_count, negative_dict

    pos_count, pos_dict, neg_count, neg_dict = returnTypeCount(train, vocab)

    pos_word = 0
    neg_word = 0

    #updating the frequencies to the probabilities
    for word in pos_dict:
        pos_word += pos_dict[word]
    for word in pos_dict:
        pos_dict[word] /= pos_word/float(100)

    for word in neg_dict:
        neg_word += neg_dict[word]
    for word in neg_dict:
        neg_dict[word] /= neg_word/float(100)

    #creating the test dataset and testing    
    test = sevenfold[i]
    test = test.reset_index()

    #predicting
    misclass = 0
    for j in range(test.shape[0]):
        pos_predict = pos_count/(pos_count + neg_count)
        neg_predict = neg_count/(pos_count + neg_count)

        for word in test['text'][j]:
            pos_predict *= pos_dict[word]
            neg_predict *= neg_dict[word]

        if(pos_predict >neg_predict):
            label = 1
        else:
            label = 0

        if test['class'][j] == label:
            continue
        else:
            misclass += 1

    accuracy[i] = misclass / test.shape[0]

print(1 - accuracy)

