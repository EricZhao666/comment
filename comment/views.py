import json
import math
import random
import pandas as pd
import requests
from collections import defaultdict
from operator import itemgetter
import json

# 获取token
from django.http import JsonResponse


def access_token():
    """"
       获取access_token
    """
    APPID = 'wxb938ce23efaa741a' # 小程序ID
    APPSECRET = '397668bbbc9849562883d33fc1488407' # 小程序秘钥
    WECHAT_URL = 'https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=' + APPID + '&secret=' + APPSECRET
    response = requests.get(WECHAT_URL)
    result = response.json()
    return result["access_token"]  # 将返回值解析获取access_token

# 查询
def databaseQuery(access_token, collection_name):
    """"
        检索数据库
       collection_name 集合的名称
       .limit() 括号内的数值限定返回的记录数
    """
    url = 'https://api.weixin.qq.com/tcb/databasequery?access_token=' + access_token
    data = {
               "env": "cloud1-0gh1hr6s3478b7cd", # 用户的数据库环境ID
    "query": "db.collection(\"" + collection_name + "\").limit(100).get()"
    }
    response = requests.post(url, data=json.dumps(data))
    result = response.json()
    # print(result) # 将返回值打印
    return result

def load_from(collection_name):
    token=access_token()
    get1 = databaseQuery(token,collection_name)
    return get1['data']

def fenge(str):
    str1=str.split(':')
    return str1[1][1:-1]
def prProcessData(orgin):
    all=[]
    for item in  orgin:
        result=item[1:-1].split(",")
        name=fenge(result[1])
        get=fenge(result[2])
        all.append([name,get])
    return all
def LoadMovieLensData(filepath, train_rate):
    ratings = pd.read_table(filepath, sep="::", header=None, names=["UserID", "MovieID", "Rating", "TimeStamp"],\
                            engine='python')
    ratings = ratings[['UserID','MovieID']]

    train = []
    test = []
    random.seed(3)
    for idx, row in ratings.iterrows():
        user = int(row['UserID'])
        item = int(row['MovieID'])
        if random.random() < train_rate:
            train.append([user, item])
        else:
            test.append([user, item])
    return PreProcessData(train), PreProcessData(test)

def PreProcessData(originData):
    """
    建立User-Item表，结构如下：
        {"User1": {MovieID1, MoveID2, MoveID3,...}
         "User2": {MovieID12, MoveID5, MoveID8,...}
         ...
        }
    """
    trainData = dict()
    for user, item in originData:
        trainData.setdefault(user, set())
        tudi=item.split(" ")
        for get in tudi:
           trainData[user].add(get)
    return trainData


class ItemCF(object):
    """ Item based Collaborative Filtering Algorithm Implementation"""
    def __init__(self, trainData, similarity="cosine", norm=True):
        self._trainData = trainData
        self._similarity = similarity
        self._isNorm = norm
        self._itemSimMatrix = dict() # 物品相似度矩阵

    def similarity(self):
        N = defaultdict(int) #记录每个物品的喜爱人数
        for user, items in self._trainData.items():
            for i in items:
                self._itemSimMatrix.setdefault(i, dict())
                N[i] += 1
                for j in items:
                    if i == j:
                        continue
                    self._itemSimMatrix[i].setdefault(j, 0)
                    if self._similarity == "cosine":
                        self._itemSimMatrix[i][j] += 1
                    elif self._similarity == "iuf":
                        self._itemSimMatrix[i][j] += 1. / math.log1p(len(items) * 1.)
        for i, related_items in self._itemSimMatrix.items():
            for j, cij in related_items.items():
                self._itemSimMatrix[i][j] = cij / math.sqrt(N[i]*N[j])
        # 是否要标准化物品相似度矩阵
        if self._isNorm:
            for i, relations in self._itemSimMatrix.items():
                max_num = relations[max(relations, key=relations.get)]
                # 对字典进行归一化操作之后返回新的字典
                self._itemSimMatrix[i] = {k : v/max_num for k, v in relations.items()}

    def recommend(self, user, N, K):
        """
        :param user: 被推荐的用户user
        :param N: 推荐的商品个数
        :param K: 查找的最相似的用户个数
        :return: 按照user对推荐物品的感兴趣程度排序的N个商品
        """
        recommends = dict()
        # 先获取user的喜爱物品列表
        items = self._trainData[user]
        for item in items:
            # 对每个用户喜爱物品在物品相似矩阵中找到与其最相似的K个
            for i, sim in sorted(self._itemSimMatrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                if i in items:
                    continue  # 如果与user喜爱的物品重复了，则直接跳过
                recommends.setdefault(i, 0.)
                recommends[i] += sim
        # 根据被推荐物品的相似度逆序排列，然后推荐前N个物品给到用户
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

    def train(self):
        self.similarity()




#    调用该函数，输入用户名，推荐数量，相近用户数量，返回推荐的土地列表
def getRecommend(name,K,N):
    origin = load_from('loadCommend')
    get = prProcessData(origin)
    result = PreProcessData(get)
    item=ItemCF(result, similarity='iuf', norm=True)
    item.train()
    return item.recommend(name,K,N)

def Recom(request,name,K,N):
    return JsonResponse({'code': 0, 'data': getRecommend(name,K,N)},
                        json_dumps_params={'ensure_ascii': False})