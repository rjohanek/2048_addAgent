- 2018.12.19
    - 决定采用分层策略
    - 突然发现 label 也要进行 one-hot 编码。。。
- 2018.12.20
    - 解决了一个困扰我好久的问题，之前不断在群里问助教。我去 Google SO 了都没用，最后偶然在搜 pytorch entropy loss 的时候偶然在 so 碰到。。。呵，人生 https://stackoverflow.com/questions/49390842/cross-entropy-in-pytorch

- 2018.12.22
    - 今天才知道我之前将棋盘onehot的方法是错误的
    - mmp 今天我决定用 Keras 了，暂时弃坑 PyTorch
- 2018.12.25
    - 这两天一直在比较各种胡乱搭建的网络的优劣，最后选定了一个稍微好点的
    - 今天在 colab 上试了下，准确率到达 61%，但是平均分数好几次都能成功到达128，可把我高兴坏了

- 2018.12.26
    - 终于搭建完了，虽说丑得不像样，但是终归是搭建好了。分阶段在线训练，扔到colab里面去让它自己跑
    - 但愿不要再出幺蛾子。。。怎么可能呢，既然有可能发生，那它一定会发生

- 2018.12.30
    - 增加了边训练边保存，改用了 LeakReLU，然后放到Colab

- 2018.1.1
    - 今天忽然发现我一直采用的并不是online training...而是在线运行的offline training
    - 发现我那种做法很不好，很浪费时间。每次训练完的数据就丢掉，然后下一轮训练时又要重新生成数据。。。而生成数据很费时间。其实之前丢掉的那些数据是可以再次使用的。。。
    - 发现自己之前设置的 (4,4,12) 真是蠢到家了。。。。

- 2018.1.2
    - 后面越来越觉得自己之前的训练过程中都大大地浪费了时间，应该尽力减小时间的浪费，想办法提高时间的利用率

- 2018.1.12
    - 明天就要交报告了，估计最高分也就是600左右了，断断续续的训练还是不行的啊
    - We don't have money, we don't have gpu...(*_*)
    - 完全用 Vim 完成的开发，可喜可贺