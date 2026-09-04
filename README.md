## 项目结构

├── demo1.py           # 完整代码：数据加载、模型、训练、评估

├── .gitignore         # git 忽略配置

├── run_result.txt     # 控制台完整运行输出日志
## 文件说明
`demo1.py`：全部业务代码，包含数据集读取、tokenizer 处理、BERT 模型构建、训练循环、评估逻辑。
`run_result.txt`：完整控制台输出，记录每一轮 epoch loss、验证集指标、测试集详细分类报告。
`.gitignore`：配置文件，忽略数据集、预训练权重、虚拟环境、日志缓存等大文件，不提交至仓库。
## 数据集信息
训练集：3000条  
验证集：1000条    
测试集：1064条   
分类类别：15条
| 类别ID | 类别名称 |
|-------|---------|
| 100 | 新闻故事 |
| 101 | 新闻文化 |
| 102 | 新闻娱乐 |
| 103 | 新闻体育 |
| 104 | 新闻财经 |
| 106 | 新闻房产 |
| 107 | 新闻汽车 |
| 108 | 新闻教育 |
| 109 | 新闻科技 |
| 110 | 新闻军事 |
| 112 | 新闻旅游 |
| 113 | 新闻国际 |
| 114 | 股票 |
| 115 | 新闻农业 |
| 116 | 新闻游戏 |

## 实验超参数
```
                第一组                  第二组                   第三组                  第四组
max_len：        64                     64                      64                      64  
batch_size：     16                     2                       2                       2
epoch：          8                      8                       6                       8
dropout：        0.1                    0.1                     0.1                     0.2
CPU训练
训练结果   CPU负载太高，实验未完成       test-acc= 81.95%         test-acc=82.99%         test-acc=81.95%
```

## 环境依赖安装

pip install torch transformers tqdm numpy swanlab


## 实验结果说明
在第二组实验时，出现了轻微过拟合。我通过减少训练轮次和增加dropout，分别做对照实验观察模型性能变化。

1. 第一组超参数设置 `batch_size=16`，受限于CPU算力，内存占用过高，电脑黑屏，实验无法完成。后续实验将batch_size下调至2，保证CPU可以正常跑完训练。
2. 第二组（batch_size=2，epoch=8，dropout=0.1）测试集准确率81.95%，训练后期验证集指标不再提升，存在轻微过拟合现象，可视化结果如下图所示：
<img width="509" height="509" alt="image" src="https://github.com/user-attachments/assets/3ee67053-e33d-4893-9230-10ffc691b41f" />

3. 第三组将训练轮次epoch由8降低到6，提前停止训练，缓解过拟合，测试集准确率提升至82.99%，效果最优。可视化结果如下图所示：
<img width="682" height="507" alt="image" src="https://github.com/user-attachments/assets/47ce6e43-cb94-4995-a2b0-1e0a0839c1ea" />

4. 第四组保持epoch=8，提升dropout至0.2，通过增大神经元随机失活抑制过拟合，用于对比正则化带来的效果。可视化结果如下图所示：
<img width="650" height="499" alt="image" src="https://github.com/user-attachments/assets/4061734b-7426-4c60-8599-0095b000a9b8" />

## 现象总结：
1. CPU训练场景下，batch_size不能设置过大，否则会造成硬件负载过高，程序崩溃黑屏。
2. 减少训练轮次（早停）能够有效缓解BERT在小数据集上的过拟合问题，可以带来测试集精度提升。
3. dropout随机失活同样可以抑制过拟合，在轮次较多时可以尝试调高dropout取值。

