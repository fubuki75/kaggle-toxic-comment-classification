# kaggle-toxic-comment-classification
比赛具体要求可见https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/leaderboard

这是东南大学-翟玉庆老师的人工智能与机器学习的课程论文，我抓住了这次机会push自己简单完成一下这个4年前的这个比赛。项目只涉及到了文本预处理、简单的基于BERT的模型构建以及API包装（假装落地），
如果是真正的比赛的话，还有很多需要做的（比如针对于正负样本不均衡的文本数据增强、模型集成等等），一步一步走吧！共勉！！

仓库文件：
* preprocess文件夹中:`api.ipynb`是基于gradio库对模型进行的简单落地实现（这个库简单玩一下即可，感觉有一些不人性的地方），`预处理.ipynb`是对文本数据进行的基于正则表达式的预处理以及
极为简单的EDA（做的太简陋了，有需要的话，一定要学习学习[这个](https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda))
* 课程论文.pdf是当时的课程论文，有对每一步的具体描述，以及最终的效果比较详细。
* model文件中，BERT是BERT+linear的预测头（即最简单的bert），效果也是最好的（跑了很长时间自己改进的模型，效果都不如原版）；bertCNN等各个版本就是bert+textCNN的魔改以及调参，效果不好，感觉卷积
操作丢失了大量的信息；bertv2是在bert的基础上，结合了点elmo的想法，想融合bert不同的层的output，效果也是一般，难顶。
