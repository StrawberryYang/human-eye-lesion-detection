# -*- coding:utf-8 -*-
from __init__ import *


this_dir = os.path.dirname(__file__)
data_path = os.path.join(this_dir, 'data/')
Encoder_Model_Path = os.path.join(this_dir, 'models/')
Svm_Model_Path = os.path.join(this_dir, 'models2/')
# print (model_path)


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
        print('save ' + path + ' success!')

def load_pkl(path):
    data = None
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        print(path + ' not exist')
    return data


def gen_model_ids(root=data_path):  # 设定模型序号便于循环
    data_names = os.listdir(root)
    f = lambda x: x[-3:]
    model_ids = list(map(f, data_names))
    return model_ids

def load_data(model_id):  # 加载训练数据
    root = './data/' + 'DataPA' + model_id  # 参考文件夹数据命名格式
    feat_mat = 'temp'  # 实际调用的值
    label_mat = 'GT'
    feats_path = root + '/A_' + model_id + '.mat'  # features
    label_path = root + '/GT_' + model_id + '.mat'  # label

    feats = sio.loadmat(feats_path)[feat_mat].T  # 将列特征，列标签转化为行的形式
    # print (feats.shape)
    label = sio.loadmat(label_path)[label_mat].T
    # print(label.shape)

    return feats, label

def load_group_data(grouped_model_id):
    feats, label = load_data(grouped_model_id[0])
    if len(grouped_model_id) > 1:
        for model_id in grouped_model_id[1:]:
            temp_feats, temp_label = load_data(model_id)
            feats = np.concatenate((feats, temp_feats), axis=0)
            label = np.concatenate((label, temp_label), axis=0)

    return feats, label

def load_OneGroupOfdata(model_names):

    model_IDs = []
    for model_name in model_names:
        model_ID = model_name[:3]
        model_IDs.append(model_ID)
    feats, label = load_group_data(model_IDs)

    print('feats shape: ', feats.shape)
    print('label shape: ', label.shape)
    return feats, label

def select_points(feats, label, num_points):
    '''
    Agrs: feats, label: grouped feats and label 
          num_points: number of selected abnormal points to train

    Return: 
    '''

    temp_feats, temp_label = feats.T, label.T
    print(temp_feats.shape, temp_label.shape)

    count = np.sum(temp_label==1)
    print ('The abnormal count is: {}'.format(count))

    n_ind = np.where(temp_label[0]==0)[0]
    a_ind = np.where(temp_label[0]==1)[0]

    num_normal, num_abnormal = len(n_ind), len(a_ind)
    # print (len(n_ind), len(a_ind))
    
    temp_n_ind = random.sample(list(n_ind), num_points)
    temp_a_ind = random.sample(list(a_ind), num_points)

    # normal_feats = temp_feats[:, n_ind][:,:20000]
    # normal_label = temp_label[:, n_ind][:,:20000]

    normal_feats = temp_feats[:, temp_n_ind]
    normal_label = temp_label[:, temp_n_ind]

    # abnormal_feats = temp_feats[:, a_ind][:,20000:0:-1]
    # abnormal_label = temp_label[:, a_ind][:,20000:0:-1]

    abnormal_feats = temp_feats[:, temp_a_ind]
    abnormal_label = temp_label[:, temp_a_ind]

    # n_ind = np.array(list(set(n_ind) - set(temp_n_ind)))
    # a_ind = np.array(list(set(a_ind) - set(temp_a_ind)))


    train_feats = np.concatenate((normal_feats, abnormal_feats), axis=1)
    train_label = np.concatenate((normal_label, abnormal_label), axis=1)
    print (train_feats.shape, train_label.shape)

    # train_feats_lst.append(train_feats.T)
    # train_label_lst.append(train_label.T)

    return train_feats.T, train_label.T


def OR(predict, label):
    temp_predict = np.array(predict)
    temp_label = np.array(label)
    predict_and_label = temp_predict & temp_label
    OR = sum(predict_and_label) / (sum(temp_predict) + sum(temp_label) - sum(predict_and_label))
    return float(OR)

def autoencoder_predict(model, feat):
    temp_feat = feat.copy()
    temp_feat = temp_feat.astype('float32') / 255.
    gx = model.predict(temp_feat)
    gx = np.array([y.argmax() for y in gx])
    gx = gx.reshape(len(gx), 1)                        
    return gx

def clf_predict(model, feat):
    temp_feat = feat.copy()
    gx = model.predict(temp_feat)
    gx = gx.reshape(len(gx), 1)
    return gx


def load_models_from_path(path):
    models = []
    files = os.listdir(path)
    # print(files)s
    for file in files:
        if file.endswith('.h'):
            model = load_model(root+file)
            models.append(model)
        elif file.endswith('.pkl'):
            model = joblib.load(root+file)
            models.append(model)
    return models

def load_models(model_names):  #调用模型
    models = []
    for model_name in model_names:
        if model_name.endswith('.h'):
            model = load_model(Encoder_Model_Path + model_name)
            models.append(model)
        elif model_name.endswith('pkl'):
            model = joblib.load(Svm_Model_Path + model_name)
    return models


def rand_group6x9():  # 随机生成模型分组
    dump_path = 'grouped_model_names.pkl'
    if os.path.exists(dump_path):
        print('grouped_model_names.pkl exist')
        with open(dump_path, 'rb') as f:
            grouped_model_names = pickle.load(f)
    else:
        print('grouped_model_names.pkl not exist')

        model_names, grouped_model_names = [], []
        for root, dirs, model_names in os.walk(model_path):
            pass
        for i in range(6):
            temp_model_names = random.sample(model_names, 9)
            model_names = list(set(model_names) - set(temp_model_names))
            grouped_model_names.append(temp_model_names)

        with open(dump_path, 'wb') as f:
            pickle.dump(grouped_model_names, f, protocol=4)

    return grouped_model_names
