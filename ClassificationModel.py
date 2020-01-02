from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras import Model
from keras.layers import Dropout, Input
from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation
from keras.layers.pooling import MaxPooling2D
import keras_metrics as km
from Metrics import Metrics
import os
import matplotlib.pyplot as plt


class ClassificationModel(object):


    def model_train(self, model,model_name,train_generator, validation_generator):
        # history = model.fit_generator(train_generator,
        #                               steps_per_epoch=8,
        #                               epochs=10,
        #                               validation_data=validation_generator,
        #                               validation_steps=20, callbacks=[metrics])
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=10,
                                      epochs=50,
                                      validation_data=validation_generator,
                                      validation_steps=5)
        print(history.history.keys())
        model.save("model/"+model_name)
        print(model_name+"模型保存成功")
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title(model_name+' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(model_name+' loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # process.estimate(history, model_name)
        # print(metrics.val_precisions)

    def inference(self, model, img_path, IMSIZE, label):
        # model = load_model("model/" + model_name)
        img = image.load_img(img_path, target_size=(IMSIZE, IMSIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.normalization(x)
        predict_result = model.predict(x)
        print(predict_result[0])
        index = np.where(predict_result[0] == np.max(predict_result[0]))
        row_number = index[0][0]
        print(label[row_number])
        return label[row_number]

    def inference_label(self, model, img_path, IMSIZE, label):
        img = image.load_img(img_path, target_size=(IMSIZE, IMSIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.normalization(x)
        predict_result = model.predict(x)
        print(predict_result[0])
        index = np.where(predict_result[0] == np.max(predict_result[0]))
        row_number = index[0][0]
        print(label[row_number])
        print(row_number)
        return row_number

    # 逻辑回归模型
    def logistic_regression_train(self, IMSIZE, train_generator, validation_generator, model_name):
        input_layer = Input([IMSIZE, IMSIZE, 3])
        x = input_layer
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(4, activation='softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.01),
                      metrics=['accuracy'])
        # 保存最好的模型
        # filepath = "weights.best.hdf5"
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
        #                              mode='max')
        # callbacks_list = [checkpoint]
        self.model_train(model, model_name, train_generator, validation_generator)

    def logistic_regression_inference(self, img_path, IMSIZE, model_name,label):
        self.inference(model_name, img_path, IMSIZE,label)


    # inceptionv3模型
    def inceptionv3_train(self, train_generator, validation_generator, model_name):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        base_model.load_weights('model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(4, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        self.model_train(model, model_name, train_generator, validation_generator)


    def inceptionv3_inference(self, img_path, IMSIZE, model_name,label):
        self.inference(model_name, IMSIZE=IMSIZE, img_path=img_path,label=label)
        return

    # alxnet模型
    def alexnet_train(self, IMSIZE, train_generator, validation_generator,model_name):
        input_layer = Input([IMSIZE, IMSIZE, 3])
        x = input_layer
        x = BatchNormalization()(x)
        x = Conv2D(96, [11, 11], strides=[4, 4], activation='relu')(x)
        x = MaxPooling2D([3, 3], strides=[2, 2])(x)
        x = Conv2D(256, [5, 5], padding="same", activation='relu')(x)
        x = MaxPooling2D([3, 3], strides=[2, 2])(x)
        x = Conv2D(384, [3, 3], padding="same", activation='relu')(x)
        x = Conv2D(384, [3, 3], padding="same", activation='relu')(x)
        x = Conv2D(256, [3, 3], padding="same", activation='relu')(x)
        x = MaxPooling2D([3, 3], strides=[2, 2])(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4, activation='softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        self.model_train(model, model_name, train_generator, validation_generator)


    def alexnet_inference(self, img_path, IMSIZE, model_name,label):
        self.inference(model_name, IMSIZE=IMSIZE, img_path=img_path,label=label)


    #lenet 模型
    def lenet_train(self, IMSIZE, train_generator, validation_generator, model_name):
        # initialize the model
        model = Sequential()
        inputShape = (IMSIZE, IMSIZE, 3)
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4))
        model.add(Activation("relu"))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        self.model_train(model, model_name, train_generator, validation_generator)


    def lenet_inference(self, img_path, IMSIZE, model_name,label):
        self.inference(model_name, IMSIZE=IMSIZE, img_path=img_path,label=label)

    # 根据测试集，计算Kappa、Jaccard、准确率，精确率，召回率，f1-score
    def evaluate(self, path, model_path, label, Dict, IMSIZE):
        model = load_model(model_path)
        y_pre = []
        y_true = []
        for root, dirs, files in os.walk(path):
            for dir in dirs:
                for root1, dirs1, files1 in os.walk(path + "/" + dir):
                    for file in files1:
                        y_true.append(Dict[dir])
                        y_pre.append(self.inference_label(model, path + "/" + dir + "/" + file, IMSIZE, label))
        me = Metrics(y_pred=y_pre, y_true=y_true)
        me.evaluate()

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    # # 根据测试集，计算准确率，精确率，召回率，f1-score
    # def estimate(self,history):
    #     # 准确率
    #     print("acc")
    #     print(history.history['acc'])
    #     print("val_acc")
    #     print(history.history['val_acc'])
    #     # 精确率
    #     print("precision")
    #     print(history.history['precision'])
    #     print("val_precision")
    #     print(history.history['val_precision'])
    #     # 召回率
    #     print("recall")
    #     print(history.history['recall'])
    #     print("val_recall")
    #     print(history.history['val_recall'])
    #     # f1_score
    #     print("f1_score")
    #     print(history.history['f1_score'])
    #     print("val_f1_score")
    #     print(history.history['val_f1_score'])
