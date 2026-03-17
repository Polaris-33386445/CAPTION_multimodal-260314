看图说话任务，通过coco2014数据集的图像-文本对，使用ResNet101提取图像特征进行编码，映射到语义空间与文本对齐后使用RNN、带注意力插件的RNN（正在尝试使用Transformer）进行解码，实现根据图像生成描述文本的多模态任务。
demo效果与注意力分数热力图如下：
![257f77af6aaac3ea3b3fec7121885d52](https://github.com/user-attachments/assets/7d48438a-9257-4608-8175-9d7490122cc7)
![04539aa823b933a726c813be71109aea](https://github.com/user-attachments/assets/60c7011c-24d9-4580-81ab-505a6874777f)
![8983c0da4b2440bb2cab01808c4a2817](https://github.com/user-attachments/assets/feb5f833-8bd2-4483-81db-aebe91f49794)
<img width="516" height="320" alt="2d79d8a078a582551b42f47fe925f935" src="https://github.com/user-attachments/assets/2c5a3f8a-abc1-4571-9792-365cf4d0de49" />
![98448425fc156662373171331fdc6ec3](https://github.com/user-attachments/assets/e1355e6a-66e7-4fc1-a0e6-126aa76bf607)
![751cfd6972682081abbf247b1f064969](https://github.com/user-attachments/assets/6b6f7817-56ab-41df-abb6-1fa7d26329d3)
![254b2d4677087a5150d253da779d21a4](https://github.com/user-attachments/assets/6ead0914-82a1-4ec5-89f6-d56347bc71d2)
![eab569c4ebf7a891c994fee38d4cd838](https://github.com/user-attachments/assets/8a711a43-2d60-4725-a945-7fcc257da3c8)
