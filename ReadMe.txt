-----------------Dependencies---------------------------
keras
tensorflow
sklearn
------------------File-------------------------------------
DownLoadPeriodImage.py：下载不同时期作品的爬虫
DownLoadWriterImage.py：下载不同作者作品的爬虫
ClassificationModel.py：模型类，包括逻辑回归、LeNet、AlexNet、Inception3V模型的训练和测试方法
Metrics.py：计算模型的性能指标
image：存训练和测试数据。	
模型测试指标结果.xlsx：模型的测试指标表格
catalog.xlsx：名画的描述数据集
test.jpg:测试用的图片
main.py:主函数入口
model: 里面存放已经训练好的模型
---------------------Usage---------------------------------
usage: main.py [-h] [--path PATH] [--model {logic,lenet,alexnet,inceptionv3}]
               [--type {writer,period}]

The argument for Writer and Period of image Prediction

optional arguments:
  -h, --help            show this help message and exit
  --path PATH, -p PATH  图片路径
  --model {logic,lenet,alexnet,inceptionv3}, -m {logic,lenet,alexnet,inceptionv3}
                        预测模型:logic,lenet,alexnet,inception
  --type {writer,period}, -t {writer,period}
                        预测时期还是作者(时期：period,作者:writer)

例子：python main.py -m inceptionv3 -t writer -p test.jpg
ps：(每次加载模型的时间会有点长,请耐心等待)