                                                         项目介绍
这是我在中国科学技术大学大一新生“科学与社会”研讨课所完成的项目：以MNIST数字集训练卷积神经网络，并结合opencv连结电脑摄像头实现对画面中手写数字的识别。

作为本人第一个github项目，可以发现，我甚至将MNIST数字集打成了MINIST数字集。如果日后我的Github仓丰富起来，那么这个小小的失误将作为我科研生涯从小白到入门的最好见证。

整体上这个项目还不成熟，也非常非常简单，但却是我迈出的第一步，让我这个目前的编程小白意识到，自己也可以学会学懂当下最“热”话题--人工智能。

如果你也一样是代码小白的话，我的经历建议你先学习一下下面五张图片，里面包括我认为初学者需要且仅需要掌握的几个概念和一些具体的数学算法。

page 1
<img width="395" height="560" alt="image" src="https://github.com/user-attachments/assets/8569d832-d450-4bb3-889d-ce0f7cc9a376" />

page 2
<img width="398" height="563" alt="image" src="https://github.com/user-attachments/assets/aa62def6-2369-432c-bb54-3aeaab7da548" />

page 3
<img width="399" height="565" alt="image" src="https://github.com/user-attachments/assets/8b5f2fbf-2ca5-4b5a-aed9-a59add3eccdd" />

page 4
<img width="400" height="563" alt="image" src="https://github.com/user-attachments/assets/2a21a71d-8d07-4820-b2bd-a5c931765e5c" />

page 5
<img width="401" height="185" alt="image" src="https://github.com/user-attachments/assets/75cd380b-2340-40d3-b6f0-ff510341c41b" />

此外可能比较有难度的是配置环境、以及下载安装所需的一些程序。你不需要在B站上学课程，我的经验说明最好的方法可以是找一个同学问问看，十分钟可以解决，或者可以直接问人工智能。但是一定要在行动中学习而不要等所有事情都准备好了再开始。

现在简单介绍一下代码： 

1、MNIST 训练卷积神经 
定义卷积神经结构如下，经两个卷积层（conv），一个池化层（pool）， 25%的dropout、两个全连接层和一个ReLU层，属于较为经典的卷积神经结构：

<img width="569" height="139" alt="image" src="https://github.com/user-attachments/assets/3af7f4ea-fa70-4dcf-89eb-4eeb818073a0" />


训练函数定义：先加载数据及对应标签（data, target），再将data模型输入model加载出output，将output和target比对得到损失(loss)，loss反向传播、计算梯度，再在优化器（本代码中选择Adam优化器）中根据设定步长调节参数，每100个batch打印一批相关参数。  

<img width="572" height="143" alt="image" src="https://github.com/user-attachments/assets/91c7d648-4170-4d57-8842-8224944f5949" />


测试函数定义：将data、target加载回device（本实验选用CPU，也可以用GPU）计算output，再使用argmax方法求得output张量对应的预测值也即pred（本实验中，pred 是数字0-9），将其与target比对后，用sum函数求和从而得到正确预测结果总数correct，从而得到并打印准确率accuracy。

<img width="573" height="167" alt="image" src="https://github.com/user-attachments/assets/18235697-5912-411e-a361-a7253c070324" />

现在进行5个epoch，每个epoch皆由上述训练函数和测试函数组成，如果测试函数输出accuracy 为最高准确率则将model保存到”best_model.pth”，并打印相应提示。 

<img width="573" height="218" alt="image" src="https://github.com/user-attachments/assets/057ca8d6-4807-432a-88d1-21d05b5c8a15" />

此外，在最初设置神经网络参数时是随机设置的，使得每次训练出的model准确率都有差异，因此手动固定所有随机种子为42，最终每次训练准确率皆为99.14%。

<img width="572" height="92" alt="image" src="https://github.com/user-attachments/assets/ae453fe2-c8e0-46a2-af72-ae85acde773b" />

2、测试卷积神经效果 
图像预处理：定义preprocess程序，将黑底白字的照片转化为28x28像素的灰度图， 从而将左图转化成MNIST风格的数字以供神经网络识别。 

<img width="654" height="274" alt="image" src="https://github.com/user-attachments/assets/2d6ca008-b3c6-4daa-9f1c-cac64d578a8d" />

结果测试：通过定义predict函数（主要思路与2.1中定义的测试函数类似，将预处理过的图像输入到model得到pred，并且呈现），结果如下，说明单次实验展现出该卷积神经效果较好。 

<img width="303" height="270" alt="image" src="https://github.com/user-attachments/assets/1026a207-1301-4d99-a5d5-efc640e1a83e" />

3、 利用OpenCV对拍摄内容进行实时数字识别 
如图所示，定义predict_from_camera()函数，通过 OpenCV 捕获摄像头画面，实时处理并显示预测结果，同样先对图像进行了颜色反转处理（白底黑字→黑底白字）以匹配 MNIST 数据集格式，再进行结果的预测与显示。同时在原来摄像头无法退出的问题基础上，新增退出代码段，以支持按Q或ESC退出功能 

<img width="575" height="289" alt="image" src="https://github.com/user-attachments/assets/79424490-463a-4066-b587-b803b5c37aae" />

预测正确时的效果如下，画面由电脑前置摄像头实时拍摄，左上绿字为预测出的数字： 

<img width="763" height="241" alt="image" src="https://github.com/user-attachments/assets/d1a3e1cb-c911-499c-8720-58f43a3f26d6" />

不过，相比2.2中的测试效果，结同摄像头的预测准确值有显著下降，推测是由于真实环境噪声较大，真实案例中前辈们有加入降噪程序，本项目中没有采用导致结果偏差明显。另一个是，有时候即便画面中并没有出现数字，程序还是会有“推测结果”的显示，这个是因为当时设计整个项目的时候没有把这点考虑进去，其实可以给预测流程追加“未识别到文字”选项，但这个还有待后续有时间继续研究。

最后，本人在做该项目之前参考过GitHub上的代码，其中我认为最新手友好的地址，希望对你有帮助：[dee1024/pytorch-captcha-recognition: 基于CNN训练的一套 "端到端" 的验证码识别模型，使用深度学习+训练数据+大量计算力，纯数字识别率高达 99.99%，数字+字母识别率 96% ](https://github.com/dee1024/pytorch-captcha-recognition)
