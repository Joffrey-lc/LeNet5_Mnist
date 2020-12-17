# 简述

三个程序分别是模型Net.py、训练主程序main.py、应用程序application.py。通过main.py训练模型并保存，通过application.py进行读取自己的手写数字并识别；三个文件夹application data 存放了我手写的10个数字，data文件夹存放了mnist数据集。mnist数据集是一个非常经典的（简单的）手写数字数据集，可以上网百度。savemodel文件夹保存了训练好的模型，方便application.py读取模型。

# 代码详解

## Net.py

![image-20201217191708899](https://mymarkdown-pic.oss-cn-chengdu.aliyuncs.com/img/image-20201217191708899.png)

## main.py

![image-20201217192203310](https://mymarkdown-pic.oss-cn-chengdu.aliyuncs.com/img/image-20201217192203310.png)

## application.py

![image-20201217192235068](https://mymarkdown-pic.oss-cn-chengdu.aliyuncs.com/img/image-20201217192235068.png)