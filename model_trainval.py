# -*- coding:utf-8 -*-
from __init__ import *
from basic import *

def autoencoder_train(model_id):   #训练并保存模型
	print('train and save model: ', model_id)
	train_feats, train_label = load_data(model_id)
	# train_feats, train_label = select_points(train_feats, train_label, num_points=8000)
	print (train_feats.shape, train_label.shape)
	train_feats = train_feats.astype('float32') / 255.  # 将特征压缩到0-1之间

	# 训练模型
	# 压缩特征维度至32维
	encoding_dim = 32

	# this is our input placeholder
	input_img = Input(shape=(200,))

	# 编码层
	encoded = Dense(200, activation='relu')(input_img)
	encoder_output = Dense(encoding_dim)(encoded)

	# 解码层
	decoded = Dense(200, activation='relu')(encoder_output)
	decoded = Dense(200, activation='tanh')(decoded)

	# 构建自编码模型
	autoencoder = Model(inputs=input_img, outputs=decoded)

	# 构建编码模型
	encoder = Model(inputs=input_img, outputs=encoder_output)

	# compile autoencoder
	autoencoder.compile(optimizer='sgd', loss='mse')

	# training
	autoencoder.fit(train_feats, train_feats, epochs=2, batch_size=200, shuffle=True)

	# 取出压缩层
	encode_output = autoencoder.layers[2].output

	# 加上softmax层
	x = Dense(2, activation="softmax")(encode_output)
	class_model = Model(inputs=input_img, outputs=x)

	# 在训练soft max的过程中，冻结第一阶段前三层的压缩参数
	#for layer in class_model.layers[:-1]:layer.trainable = False

	# 训练
	rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0001)
	class_model.compile(optimizer=rmsprop,
						loss='sparse_categorical_crossentropy',
						metrics=['accuracy'])
	print('Training ------------')
	# Another way to train the model
	class_model.fit(train_feats, train_label, epochs=2, batch_size=200)

	model_save_path = model_path + model_id + '.h'
	class_model.save(model_save_path)

def svm_train(model_id):
	print('train and save model: ', model_id)
	train_feats, train_label = load_data(model_id)
	# train_feats, train_label = select_points(train_feats, train_label, num_points=300000)

	# pca = decomposition.PCA(n_components=20)
	# pca.fit(train_feats)
	# train_feats = pca.transform(train_feats)

	print (train_feats.shape, train_label.shape)
	# print (type(train_label.T))
	clf = LinearSVC()
	clf = clf.fit(train_feats, train_label.ravel())
	model_save_path = model_path + model_id + '.pkl'
	joblib.dump(clf, model_save_path, compress=1)

def model_evaluate(model_name): # for all data
	data_ids = gen_model_ids(root=data_path)
	ORs = []
	# predict = model.predict(train_feats)
	for data_id in data_ids:
		feats, label = load_data(data_id)
		if model_name.endswith('.h'):
			model = load_model(model_name)
			predict = autoencoder_predict(model, feats)
		elif model_name.endswith('.pkl'):
			model = joblib.load(model_name)
			predict = clf_predict(model, feats)

		iou = OR(predict.astype(int), label.astype(int))
		print (model_name + ' test ' + 'data' + data_id + ': ' + str(iou))
		ORs.append(iou)

	print ('OR mean: ', np.mean(ORs))

if __name__ == '__main__':
	model_ids = gen_model_ids(root=data_path)
	for model_id in model_ids:
		svm_train(model_id)
	# model_evaluate('./models2/501.pkl')



