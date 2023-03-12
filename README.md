# WeUSM (Develop)

###  技术趋同 

语音识别目前技术框架趋同：ctc transducer 或者较新cif base， 这几个主流的方案其实做的事情是一致的：

1 对声学信号进行特征提取

2 损失函数“对齐到文本” 建模 

对于第一点，目前可以分为两种发案：a 无监督 b 有监督

对于第二点，ctc/transducer/cif 有各自的对齐拓扑， 并且在有监督训练过程中 将部分语意信息带入
到: 1 encoder 上层（ctc）2 predictor ILM(transducer) 3 cif decoder 或者cif-transducer predcior 中

###  兼顾signal 和 text 

要建立强悍的识别模型，必须要同时考虑到signal 和 text 两种信息融合于模型中， 工业级端到端识别系统往往
上万级乃至几十万积的数据覆盖率较多信号和文本，但是依旧会存在各种厂尾问题，比如场景更换带来信号的突变，异或者领域切换
导致文本不匹配，前者往往靠各种数据增强，而后者各种外挂语言模型

近些年各种无/自监督方法（wav2vec w2vbert hubert bestrq等）在信号层面上提升模型的表征（representation）能力
但是对于asr的文本依旧是采用有监督的方式带入 

### 大厂最新 

谷歌的USM 基于之前一系列工作，提供一种新的方式
1 信号层面依旧是自监督起步，但是需要建模训练方法足够简单
2 海量文本text（unpair） 需要在自监督/有监督过程中同时建模
3 海量文本需要模块生成语音中间表征，形成（text, audio_dummy 进行伪有监督

这三部从直觉上，有望解决：1 信号建模 2 文本融入 3 文本和信号会有海量piar(类似人看书时往往会有默念行为 再次听别人说大概能秒懂)

### 本仓库目的：尝试复现谷歌USM论文中pipeline，主要实现：

- [ ] bestrq pretrain
- [ ] unpiar data training (inject text)
- [ ] text to speech representation

使用工具：wenet toolkit， 会把相关paper定时整理到该README中
