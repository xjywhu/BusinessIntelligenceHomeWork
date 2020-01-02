from keras.preprocessing.image import ImageDataGenerator
from ClassificationModel import ClassificationModel
from matplotlib import pyplot as plt
from keras.models import load_model
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import argparse

IMSIZE = 299
parser = argparse.ArgumentParser(description='The argument for Writer and Period of image Prediction')
parser.add_argument('--path', '-p', type=str, help='图片路径', default="test.jpg",)
parser.add_argument('--model', '-m', type=str, choices=["logic", "lenet", "alexnet", "inceptionv3"], help='预测模型:logic,lenet,alexnet,inception', default="inceptionv3")
parser.add_argument('--type', '-t', type=str, choices=["writer", "period"], help='预测时期还是作者(时期：period,作者:writer)', default='writer')
def train_and_evaluate():
    # 作者识别DataGenerator
    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.5,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True).flow_from_directory(
        'image/writer/train',
        target_size=(IMSIZE, IMSIZE),
        batch_size=50,  # 作者识别有500张，
        class_mode='categorical')

    validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'image/writer/validation',
        target_size=(IMSIZE, IMSIZE),
        batch_size=20,  # 作者识别有100张
        class_mode='categorical')

    # 时期识别DataGenerator

    train_period_generator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.5,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True).flow_from_directory(
        'image/period/train',
        target_size=(IMSIZE, IMSIZE),
        batch_size=10,  # 时期识别有100张(每类)
        class_mode='categorical')

    validation_period_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'image/period/validation',
        target_size=(IMSIZE, IMSIZE),
        batch_size=20,  # 时期识别有20张(每类)
        class_mode='categorical')

    # Model = ClassificationModel()
    # # 图片时期预测模型的训练和评估
    # Model.logistic_regression_train(IMSIZE, train_period_generator, validation_period_generator, "logic_period_inference")
    # Model.lenet_train(IMSIZE, train_period_generator, validation_period_generator, "lenet_period_inference")
    # Model.alexnet_train(IMSIZE, train_period_generator, validation_period_generator, "alexnet_period_inference")
    # Model.inceptionv3_train(train_period_generator, validation_period_generator, "inceptionv3_period_inference")
    #
    # Model.evaluate("image/period/validation", "model/lenet_period_inference", label2, Dict2, IMSIZE)
    # Model.evaluate("image/period/validation", "model/inceptionv3_period_inference", label2, Dict2, IMSIZE)
    # Model.evaluate("image/period/validation", "model/logic_period_inference", label2, Dict2, IMSIZE)
    # Model.evaluate("image/period/validation", "model/alexnet_period_inference", label2, Dict2, IMSIZE)
    #
    # # 图片作者预测模型的训练和评估
    # Model.logistic_regression_train(IMSIZE,train_generator,validation_generator,"logic_writer_inference")
    # Model.lenet_train(IMSIZE, train_generator, validation_generator, "lenet_writer_inference")
    # Model.alexnet_train(IMSIZE, train_generator, validation_generator, "alexnet_writer_inference")
    # Model.inceptionv3_train(IMSIZE, train_generator, validation_generator, "inceptionv3_writer_inference")
    #
    # Model.evaluate("image/writer/validation", "model/logic_writer_inference", label1, Dict1, IMSIZE)
    # Model.evaluate("image/writer/validation", "model/inceptionv3_writer_inference", label1, Dict1, IMSIZE)
    # Model.evaluate("image/writer/validation", "model/lenet_writer_inference", label1, Dict1, IMSIZE)
    # Model.evaluate("image/writer/validation", "model/alexnet_writer_inference", label1, Dict1, IMSIZE)
    return


def show_image(path, label):
    lena = mpimg.imread(path)  # 读取和代码处于同一目录下的 lena.png
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
    # lena.shape  # (512, 512, 3)
    plt.imshow(lena)  # 显示图片
    plt.text(0.5, 1, label)
    plt.axis('off')  # 不显示坐标轴
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    label1 = {0: 'Leonardo', 1: 'Monet', 2: 'Picasso', 3: 'VanGogh'}
    Dict1 = {'Leonardo': 0, 'Monet': 1, 'Picasso': 2, 'VanGogh': 3}
    label2 = {0: 'After Renaissance', 1: 'Middle', 2: 'Modern', 3: 'Renaissance'}
    Dict2 = {'After Renaissance': 0, 'Middle': 1, 'Modern': 2, 'Renaissance': 3}
    # model_writer_dic = {"logic": "model/logic_writer_inference",
    #                     "lenet": "model/lenet_writer_inference",
    #                     "alexnet": "model/alexnet_writer_inference",
    #                     "inceptionv3": "model/inceptionv3_writer_inference"}
    #
    # model_period_dic = {"logic": "model/logic_period_inference",
    #                     "lenet": "model/logic_period_inference",
    #                     "alexnet": "model/alexnet_period_inference",
    #                     "inceptionv3": "model/inceptionv3_period_inference"}
    print("正在加载模型...")
    model11 = load_model("model/logic_writer_inference")
    # model12 = load_model("model/alexnet_writer_inference")
    # model13 = load_model("model/logic_writer_inference")
    model14 = load_model("model/inceptionv3_writer_inference")

    model21 = load_model("model/logic_period_inference")
    # model22 = load_model("model/alexnet_period_inference")
    # model23 = load_model("model/logic_period_inference")
    model24 = load_model("model/inceptionv3_period_inference")

    model_writer_dict = {'logic': model11, 'lenet': model14, 'alexnet': model14, 'inceptionv3': model14}
    model_period_dict = {'logic': model21, 'lenet': model24, 'alexnet': model24, 'inceptionv3': model24}
    print("模型加载完毕...")
    #
    Model = ClassificationModel()
    if args.type == "writer":
        label = "writer:" + Model.inference(img_path=args.path, model=model_writer_dict[args.model], IMSIZE=IMSIZE, label=label1)
    else:
        label = "period:" + Model.inference(img_path=args.path, model=model_period_dict[args.model], IMSIZE=IMSIZE, label=label2)
    show_image(args.path, label)

