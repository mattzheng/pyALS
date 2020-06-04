
'''
ALS - 协同过滤

功能一：协同过滤进行商品推荐

'''
from ALS.utils import run_time

def load_movie_ratings(path):
    """Load movie ratings data for recommedation.
    Returns:
        list -- userId, movieId, rating
    """

    #file_name = "movie_ratings"
    #path = os.path.join(BASE_PATH, "dataset", "%s.csv" % file_name)
    f = open(path)
    lines = iter(f)
    col_names = ", ".join(next(lines)[:-1].split(",")[:-1])
    print("The column names are: %s." % col_names)
    data = [[float(x) if i == 2 else int(x)
             for i, x in enumerate(line[:-1].split(",")[:-1])]
            for line in lines]
    f.close()

    return data

def format_prediction(item_id, score):
    return "item_id:%d score:%.2f" % (item_id, score)


'''
ALS - 协同过滤

功能二:人群放大

'''
from tqdm import tqdm
import numpy as np
import gensim

def wordvec_save2txt(vocab_dict_word2vec,save_path = 'Tencent_to_cec.txt',encoding = 'utf-8-sig'):
    # 保存下来的词向量(字典型)，存储成为.txt格式
    '''
    input:
        dict,{word:vector}
    '''
    length = len(vocab_dict_word2vec)
    size = len(list(vocab_dict_word2vec.values())[0])
    f2 = open(save_path,'a')
    f2.write('%s %s\n'%(length,size))
    f2.close()

    for keys_,values_ in vocab_dict_word2vec.items():
        f2 = open(save_path,'a',encoding =encoding)
        f2.write(str(keys_) + ' ' + ' '.join([str(i) for i in list(values_)]) + '\n')
        f2.close()
    pass


def coverage(item_a,item_b,model):
    '''
    覆盖率 = item_a  / (item_a + item_b)
    item_a看电影的ID，item_b看电影的ID，群体的重叠计数
    '''
    
    return len(model.user_items[item_a] & model.user_items[item_b]) / len(model.user_items[item_a] )

    
if __name__ == '__main__':

    '''
    ALS - 协同过滤
    
    功能一：协同过滤进行商品推荐
    
    '''
    # 加载数据
    path = 'data/movie_ratings.csv'
    X = load_movie_ratings(path) # 100836
    # [[1, 1, 4.0], [1, 3, 4.0], [1, 6, 4.0], [1, 47, 5.0], [1, 50, 5.0]]  
    # (用户ID，购物ID，评分)

    # 训练模型
    from ALS.pyALS import ALS
    model = ALS()
    model.fit(X, k=20, max_iter=2)
    
    # 商品推荐
    print("Showing the predictions of users...")
    # Predictions
    user_ids = range(1, 5)
    predictions = model.predict(user_ids, n_items=2)
    for user_id, prediction in zip(user_ids, predictions):
        _prediction = [format_prediction(item_id, score)
                       for item_id, score in prediction]
        print("User id:%d recommedation: %s" % (user_id, _prediction))
    # User id:1 recommedation: ['item_id:858 score:4.06', 'item_id:2762 score:3.29']
    # User id:2 recommedation: ['item_id:4993 score:0.75', 'item_id:527 score:0.69']
    # User id:3 recommedation: ['item_id:2959 score:0.28', 'item_id:4995 score:0.24']
    # User id:4 recommedation: ['item_id:1210 score:2.07', 'item_id:50 score:1.88']
        
    # 模型保存与加载
    import pickle
    pickle.dump(model,open('model/movie.model','wb')   )  
    model = pickle.load(open('model/movie.model', 'rb'))
        
        
    '''
    ALS - 协同过滤
    
    功能二:人群放大
    
    '''
    # 将用户矩阵+商品矩阵，像word2vec一样进行保存
    user_matrix = np.array(model.user_matrix.data)
    item_matrix = np.array(model.item_matrix.data)
    print(user_matrix.shape,item_matrix.shape) # ((20, 610), (20, 9724))
    
    user_embedding = {model.user_ids[n]:user_matrix.T[n] for n in range(len(model.user_ids))}
    item_embedding = {model.item_ids[n]:item_matrix.T[n] for n in range(len(model.user_ids))}
    
    
    wordvec_save2txt(user_embedding,save_path = 'w2v/user_embedding_10w_50k_10i.txt',encoding = 'utf-8-sig')
    wordvec_save2txt(item_embedding,save_path = 'w2v/item_embedding_10w_50k_10i.txt',encoding = 'utf-8-sig')
    
    
    # 向量载入
    embedding = gensim.models.KeyedVectors.load_word2vec_format('w2v/user_embedding_10w_50k_10i.txt',binary=False)
    embedding.init_sims(replace=True)  # 神奇，很省内存，可以运算most_similar
    
    # 向量求相似
    item_a = 1
    simi = embedding.most_similar(str(item_a), topn=50)
    #[('79', 0.9031778573989868),
    # ('27', 0.882379412651062)]
    
    # 不同item之间人群重合度
    item_a = 1
    item_b = 39
    coverage(item_a,item_b,model)

