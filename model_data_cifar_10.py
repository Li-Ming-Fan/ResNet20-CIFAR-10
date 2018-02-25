# -*- coding: utf-8 -*-



data_dir = './cifar-10-batches-py'

#
# cifar-10： ‘./cifar-10-batches-py’
#
# data_batch_1, 2, 3, 4, 5
# test_batch
#

#
# cifar-100： ‘./cifar-100-python’
#
# train
# test
#

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    

def z_convert_data(data):
    #
    import numpy as np
    data = np.float64(data) / 255
    data = data.reshape((-1,3,32,32))  # row major: hight, width
    data = np.transpose(data, (0,2,3,1))
    #
    return np.float32(data)
    #
    
def z_convert_labels(labels):
    #
    # from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    #
    labels = [[v] for v in labels]
    ohe = OneHotEncoder()
    ohe.fit(labels)
    labels = ohe.transform(labels).toarray()
    #
    import numpy as np
    return np.float32(labels)
    #
    
def load_train_data():
    #
    data_train = {'x':[], 'y':[]}
    #
    for i in range(5):
        #
        data_dict = unpickle(data_dir +'/data_batch_' + str(i+1))
        #
        data = data_dict[b'data']       # uint8, (10000, 3072)
        labels = data_dict[b'labels']   # 0-9, list, 10000
        #
        data = z_convert_data(data)
        labels = z_convert_labels(labels)
        #
        data_train['x'].extend(data)
        data_train['y'].extend(labels)
        #
    #
    return data_train
    #
   
def load_test_data():
    #
    data_test = {'x':[], 'y':[]}
    #
    data_dict = unpickle(data_dir +'/test_batch')
    #
    data = data_dict[b'data']       # uint8, (10000, 3072)
    labels = data_dict[b'labels']   # 0-9, list, 10000
    #
    data = z_convert_data(data)
    labels = z_convert_labels(labels)
    #
    data_test['x'].extend(data)
    data_test['y'].extend(labels)    
    #
    return data_test
    #
    
def draw_images(data, list_draw, save_dir):
    #
    from PIL import Image
    #
    for index in list_draw:
        #
        # image
        r = Image.fromarray(data['x'][index][:,:,0] * 255).convert('L')
        g = Image.fromarray(data['x'][index][:,:,1] * 255).convert('L')
        b = Image.fromarray(data['x'][index][:,:,2] * 255).convert('L')
        #
        file_target = str(index)+'.png'
        img_target = Image.merge("RGB", (r, g, b))
        img_target.save(file_target)
        #
    #
    
    
if __name__ == '__main__':
    #
    import numpy as np
    #
    data = load_test_data()
    #
    TYPE = 8
    RANGE = 100
    for i in range(RANGE):
        #
        if np.argmax(data['y'][i]) == TYPE:
            #
            draw_images(data, [i], './')
            #
        #
    pass

#
# test_dict = unpickle(data_dir +'/test_batch')
#
# print(test_dict.keys())
# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
# print(len(test_dict[b'labels']))
# print(test_dict[b'labels'][0])  # not one-hot
# print(test_dict[b'batch_label']) 
#
# data = test_dict[b'data']       # uint8
# labels = test_dict[b'labels']   # 0-9
#
# print(len(data)         # 10000 
# print(len(labels))
#
