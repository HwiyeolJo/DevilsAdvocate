import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm_notebook
import torch

""" Data Reader """

def LoadDatasets(DataName):
    
    if DataName == "DBpedia":
        names = ["class", "title", "content"]
        train_csv = pd.read_csv("../../Data/ClfDataset/DBpedia/train.csv", names=names)
        test_csv = pd.read_csv("../../Data/ClfDataset/DBpedia/test.csv", names=names)
        shuffle_csv = train_csv.sample(frac=1)
        x_train = pd.Series(shuffle_csv["content"])
        y_train = pd.Series(shuffle_csv["class"])
        y_train = y_train - 1# The Index starts from 1
        x_test = pd.Series(test_csv["content"])
        y_test = pd.Series(test_csv["class"])
        y_test = y_test - 1

        f = open("../../Data/ClfDataset/DBpedia/classes.txt")

        TopicList = {}
        for line in f:
            TopicList[line.strip()] = len(TopicList)

        Idx2Topic = dict(zip(TopicList.values(), TopicList.keys()))
        Idx2Topic_list = []
        for i in range(len(set(TopicList))):
            Idx2Topic_list.append(Idx2Topic[i])
        Idx2Topic_list = np.array(Idx2Topic_list)
        
        return x_train, y_train, [], [], x_test, y_test, TopicList, Idx2Topic
    
    elif DataName == "YahooAnswersUpper":
        FileList = os.listdir("../../Data/ClfDataset/YahooAnswer")
        TopicList = {}; x = []; y = [];

        for f in FileList:
            topic = f.split('.')[0] # [0] for Super-category / [1] for Sub-category
            if topic not in TopicList.keys():
                TopicList[topic] = len(TopicList)
            # Read Data
            FileObject = open("../../Data/ClfDataset/YahooAnswer/" + f, 'r')
            text = FileObject.read()
            text = re.findall(r"<TEXT>(.+?)</TEXT>", text, re.DOTALL)
            for t in text:
                x.append(t.strip())
                y.append(topic)    

        # Indexing Y
        for i in range(len(y)):
            y[i] = TopicList[y[i]]

        Idx2Topic = dict(zip(TopicList.values(), TopicList.keys()))
        Idx2Topic_list = []
        for i in range(len(set(TopicList))):
            Idx2Topic_list.append(Idx2Topic[i])
        Idx2Topic_list = np.array(Idx2Topic_list)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        
        return x_train, y_train, [], [], x_test, y_test, TopicList, Idx2Topic
    
    elif DataName == "YahooAnswersLower":
        FileList = os.listdir("../../Data/ClfDataset/YahooAnswer/")
        TopicList = {}; x = []; y = [];

        for f in FileList:
            topic = f.split('.')[1] # [1] for Super-category / [1] for Sub-category
            if topic not in TopicList.keys():
                TopicList[topic] = len(TopicList)
            # Read Data
            FileObject = open("../../Data/ClfDataset/YahooAnswer/" + f, 'r')
            text = FileObject.read()
            text = re.findall(r"<TEXT>(.+?)</TEXT>", text, re.DOTALL)
            for t in text:
                x.append(t.strip())
                y.append(topic)    

        # Indexing Y
        for i in range(len(y)):
            y[i] = TopicList[y[i]]

        Idx2Topic = dict(zip(TopicList.values(), TopicList.keys()))
        Idx2Topic_list = []
        for i in range(len(set(TopicList))):
            Idx2Topic_list.append(Idx2Topic[i])
        Idx2Topic_list = np.array(Idx2Topic_list)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        
        return x_train, y_train, [], [], x_test, y_test, TopicList, Idx2Topic
    
    elif DataName == "YahooAnswerv2":
        x_train, y_train, x_test, y_test = [], [], [], []
        x_valid, y_valid = [], []

        FileObject = open("../../Data/ClfDataset/YahooAnswerv2/train.csv", 'r', encoding='utf-8')
        for line in FileObject:
            # line = line.split(',')
            x_train.append(line[4:][1:-1])
            y_train.append(int(line[1])-1)
        FileObject.close()
        
        FileObject = open("../../Data/ClfDataset/YahooAnswerv2/test.csv", 'r', encoding='utf-8')
#         FileObject.readline()
        for line in FileObject:
            # line = line.split(',')
            x_test.append(line[4:][1:-1])
            y_test.append(int(line[1])-1)
        FileObject.close()
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_valid, y_valid = np.array(x_valid), np.array(y_valid)
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        TopicList = { }
        Idx2Topic =  { }
        Idx2Topic_list = []
        for k in (Idx2Topic.keys()):
            Idx2Topic_list.append(Idx2Topic[k])
        Idx2Topic_list = np.array(Idx2Topic_list)
        
        return x_train, y_train, x_valid, y_valid, x_test, y_test, TopicList, Idx2Topic
    
    elif DataName == "YelpReviews":
        x_train, y_train, x_test, y_test = [], [], [], []
        x_valid, y_valid = [], []

        FileObject = open("../../Data/ClfDataset/YelpReviews/train.csv", 'r')
        for line in FileObject:
            # line = line.split(',')
            x_train.append(line[4:][1:-1])
            y_train.append(int(line[1])-1)
        FileObject.close()
        
        FileObject = open("../../Data/ClfDataset/YelpReviews/test.csv", 'r')
        FileObject.readline()
        for line in FileObject:
            # line = line.split(',')
            x_test.append(line[4:][1:-1])
            y_test.append(int(line[1])-1)
        FileObject.close()
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_valid, y_valid = np.array(x_valid), np.array(y_valid)
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        TopicList = { "very_negative": 0, "negative": 1, "neutral": 2, "positive": 3, "very_positive": 4 }
        Idx2Topic =  { 0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive" }
        Idx2Topic_list = []
        for k in (Idx2Topic.keys()):
            Idx2Topic_list.append(Idx2Topic[k])
        Idx2Topic_list = np.array(Idx2Topic_list)
        
        return x_train, y_train, x_valid, y_valid, x_test, y_test, TopicList, Idx2Topic
        
    elif DataName == "YelpReviewsPolarity":
        x_train, y_train, x_test, y_test = [], [], [], []
        x_valid, y_valid = [], []

        FileObject = open("../../Data/ClfDataset/YelpReviews/train.csv", 'r')
        for line in FileObject:
            # line = line.split(',')
            x_train.append(line[4:][1:-1])
            y_val = int(line[1])-1
            if y_val <= 1: y_val = 0
            elif y_val == 2: y_val = 1
            elif y_val >= 3: y_val = 2
            y_train.append(y_val)
        FileObject.close()
        
        FileObject = open("../../Data/ClfDataset/YelpReviews/test.csv", 'r')
        FileObject.readline()
        for line in FileObject:
            # line = line.split(',')
            x_test.append(line[4:][1:-1])
            y_val = int(line[1])-1
            if y_val <= 1: y_val = 0
            elif y_val == 2: y_val = 1
            elif y_val >= 3: y_val = 2
            y_test.append(y_val)
        FileObject.close()
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_valid, y_valid = np.array(x_valid), np.array(y_valid)
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        TopicList = { "negative": 0, "neutral": 1, "positive": 2}
        Idx2Topic =  { 0: "negative", 1: "neutral", 2: "positive"}
        Idx2Topic_list = []
        for k in (Idx2Topic.keys()):
            Idx2Topic_list.append(Idx2Topic[k])
        Idx2Topic_list = np.array(Idx2Topic_list)
        
        return x_train, y_train, x_valid, y_valid, x_test, y_test, TopicList, Idx2Topic
        
    elif DataName == "IMDB":
        x_train, y_train, x_test, y_test = [], [], [], []
        x_valid, y_valid = [], []

        FileObject = open("../../Data/ClfDataset/IMDB/labeledTrainData.tsv", 'r', encoding='utf-8')
        FileObject.readline()
        for line in FileObject:
            line = line.split('\t')
            x_train.append(line[2][1:-1])
            y_train.append(int(line[1]))
        FileObject.close()
        
        FileObject = open("../../Data/ClfDataset/IMDB/testData.tsv", 'r', encoding='utf-8')
        FileObject.readline()
        for line in FileObject:
            line = line.split('\t')
            x_test.append(line[1][1:-1])
            y_val = int(line[0].split('_')[-1][:-1])
            y_test.append(1 if y_val > 5 else 0)
        FileObject.close()
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_valid, y_valid = np.array(x_valid), np.array(y_valid)
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        TopicList = { "negative": '0', "positive": '1'}
        Idx2Topic =  { 0: "negative", 1: "positive"}
        Idx2Topic_list = []
        for k in (Idx2Topic.keys()):
            Idx2Topic_list.append(Idx2Topic[k])
        Idx2Topic_list = np.array(Idx2Topic_list)
        
        return x_train, y_train, x_valid, y_valid, x_test, y_test, TopicList, Idx2Topic
    
    elif DataName == "IMDBv2":
        x_train, y_train, x_test, y_test = [], [], [], []
        x_valid, y_valid = [], []

        FileObject = open("../../Data/ClfDataset/IMDBv2/imdb.train.txt.ss", 'r', encoding='utf-8')
        for line in FileObject:
            line = line.split('\t')
            x_train.append(line[6])
            y_train.append(int(line[4]))
        FileObject.close()
        
        FileObject = open("../../Data/ClfDataset/IMDBv2/imdb.dev.txt.ss", 'r', encoding='utf-8')
        for line in FileObject:
            line = line.split('\t')
            x_valid.append(line[6])
            y_valid.append(int(line[4]))
        FileObject.close()
        
        FileObject = open("../../Data/ClfDataset/IMDBv2/imdb.test.txt.ss", 'r', encoding='utf-8')
        for line in FileObject:
            line = line.split('\t')
            x_test.append(line[6])
            y_test.append(int(line[4]))
        FileObject.close()
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_valid, y_valid = np.array(x_valid), np.array(y_valid)
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        TopicList = {}
        Idx2Topic =  {}
        
        return x_train, y_train, x_valid, y_valid, x_test, y_test, TopicList, Idx2Topic
    
    elif DataName == "AGNews":
        names = ["class", "title", "content"]
        train_csv = pd.read_csv("../../Data/ClfDataset/AGNews/train.csv", names=names)
        test_csv = pd.read_csv("../../Data/ClfDataset/AGNews/test.csv", names=names)
        shuffle_csv = train_csv.sample(frac=1)
        x_train = pd.Series(shuffle_csv["content"])
        y_train = pd.Series(shuffle_csv["class"])
        y_train = y_train - 1 # The Indice start from 1
        x_valid, y_valid = [], []
        x_test = pd.Series(test_csv["content"])
        y_test = pd.Series(test_csv["class"])
        y_test = y_test - 1

        f = open("../../Data/ClfDataset/AGNews/classes.txt")

        TopicList = {}
        for line in f:
            TopicList[line.strip()] = len(TopicList)

        Idx2Topic = dict(zip(TopicList.values(), TopicList.keys()))
        Idx2Topic_list = []
        for i in range(len(set(TopicList))):
            Idx2Topic_list.append(Idx2Topic[i])
        Idx2Topic_list = np.array(Idx2Topic_list)
        
        return x_train, y_train, x_valid, y_valid, x_test, y_test, TopicList, Idx2Topic

""" Data Preprocessing """

WordDict = {"<NONE>":0, "<OOV>":1};

def Preprocessing(string):
    string = re.sub(r"([\-,;\.!\?:\'\"/\|_#\$%\^\&\*~`\+=<>\(\)\[\]\{\}])", " \\1 ", string)
    string = re.sub("<sssss>", '', string)
    return string.lower()

def DataProcessing(data, WordDict, WordCnt, TrainFlag):
    datas = []; MaxSeqLen = 0
    pbar = tqdm_notebook(total = len(data))
    for cnt, sent in enumerate(data):
        pbar.update(1)
        sent = Preprocessing(sent)
        Token = sent.split()
        # MaxSeqLen Constraint
#         Token = Token[:100]
        if TrainFlag:
            for T in Token:
                if T not in WordDict:
                    WordDict[T] = len(WordDict)
                    WordCnt[T] = 1
                else:
                    WordCnt[T] = WordCnt[T]+1
                    
            if len(Token) > MaxSeqLen:
                MaxSeqLen = len(Token)
            datas.append(Token)
        else:
            datas.append(Token)
    pbar.close()
    return datas, WordDict, WordCnt, MaxSeqLen

def Word2Tensor(Tokens, WordDict, MaxSeqLen):
    IdxTensor = torch.zeros(MaxSeqLen).long()
    for i, t in enumerate(Tokens):
        if i >= MaxSeqLen: break
        try:
            IdxTensor[i] = WordDict[t]
        except KeyError:
            IdxTensor[i] = WordDict['<OOV>']
    return IdxTensor

def EmbeddingNumpy(Data, WordDict, MaxSeqLen):
    Embed_np = np.zeros([len(Data), MaxSeqLen])
    pbar = tqdm_notebook(total = len(Data))
    for i in range(len(Data)):
        pbar.update(1)
        Embed_np[i,:] = Word2Tensor(Data[i], WordDict, MaxSeqLen)
    pbar.close()
    return Embed_np