ML Final - TV conversation
===

## Inroduction & Motivation (1%)
我們這次選擇的題目是tv conversation, 主要是給定一段中文電視劇的對話後，從六個選項中選出最適合的回應。選擇這個題目主要是因為雖然machine learning運用在自然語言處理上已行之有年，但由於分詞難度或語言普及程度等等的關係，中文的文本處理相較於英文來說，相對資源較少，難度也更高。在認為這題目充滿挑戰性的情況下，我們最終決定選擇它作為final project的題目。

## Data Preprocessing/Feature Engineering (2%)
### Simple Text Replacement
用正則表達式函式庫, ```re``` 將以下文字做處理。 
```python
pat_remove = re.compile('["\-()]')
pat_punc = re.compile('[，、？\.+]')
```
其中```pat_remove```會被移除，```pat_punc```會被轉成單格空白。
### Jieba
將一句話拆乘一個一個詞，並往後以詞作為一個有意義的單位。
### Pretrain CBOW & skip-gram
|CBOW|skip-gram|
|-|-|
|![](https://i.imgur.com/MkPnkrs.png)|![](https://i.imgur.com/vPuiYt2.png)|
我們分別使用skip-gram和CBOW對training data做word embedding。
並在接下來訓練模型的時候，會將文字利用上述兩種pretrain-model轉換成長度較為固定的vector。
### \" special case
我們觀察testing data時，發現有\"的句子通常都是歌詞的一開始，而在對話中較少有一句正常的話接歌曲。因此我們會將這些選項刪除，使干擾選項減少，進而增加正確率。

## Model Description (4%)
### mlstm Model

#### pretrain
word embedding 用 word2vec(skipgram) pretrain

#### training
從每個句子的下三句裡面選出一句當成正確答案，另外 sample 五句當成錯誤答案。把題目和選項 encode 之後做內積，然後 6 個內積值取 softmax，loss 用 cross entropy。
optimizer 用 Adam，learning rate 從 5e-4 線性下降到 0，訓練 300 個 epoch。
（訓練時內積換成 `element-wise product -> dropout(0.5) -> sum`）



#### encoder
|Layer|Params|
|-|-|
|word embedding|256d|
|dropout|p=0.5|
|mlstm|取最後的c作為output|

#### mlstm cell:

$m_t = (W_{hm} h_{t-1} + b_{hm}) \odot (W_{im} x_t + b_{im})$
$i_t = \sigma(W_{ii} x_t + b_{ii} + W_{mi} m_t + b_{mi})$
$f_t = \sigma(W_{if} x_t + b_{if} + W_{mf} m_t + b_{mf})$
$g_t = \tanh(W_{ig} x_t + b_{ig} + W_{mg} m_t + b_{mg})$
$o_t = \sigma(W_{io} x_t + b_{io} + W_{mo} m_t + b_{mo})$
$c_t = f_t c_{t-1} + i_t g_t$
$h_t = o_t \tanh(c_t)$

所有 W 都加 weight normalization

### BOW Model

#### pretrain
word embedding 用 word2vec(CBOW) pretrain

#### training
從每個句子的下一句裡面選出一句當成正確答案，另外 sample 一句當成錯誤答案。把兩句做內積之後 sigmoid，loss 用 binary cross entropy。
optimizer 用 Adamax
* $lr = 0.002$
* $\beta_1 = 0.9$
* $\beta_2 = 0.999$
* $decay= 1e-5$

訓練 15 個 epoch。

（訓練時內積換成 `sum -> Dot -> Dense(1,sigmoid)`）

#### encoder
|Layer|Params|
|-|-|
|word embedding|64d|
|Dense|1024|
|Dropout|0.2|
|LeakyReLU|0.1|
|BatchNormalization|\<keras default\>|
|Dense|4096|
|Dropout|0.3|
|LeakyReLU|0.1|
|BatchNormalization|\<keras default\>|

### dual encoder Model
#### pretrain
word embedding 用 word2vec(CBOW) pretrain

#### training



## Experiment and Discussion (6%)


## Conclusion (1%)

## Reference (1%)
* Multiplicative LSTM for sequence modelling, 
https://arxiv.org/pdf/1609.07959.pdf
* An efficient framework for learning sentence representations
https://arxiv.org/pdf/1803.02893.pdf
* Lecun uniform initializers
http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
* Adamax/Adam Optimizers
https://arxiv.org/pdf/1412.6980v8.pdf
* Software Framework for Topic Modelling with Large Corpora
https://radimrehurek.com/gensim/lrec2010_final.pdf
* Dropout:  A Simple Way to Prevent Neural Networks from Overfitting
http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
* Empirical Evaluation of Rectified Activations in Convolution Network
https://arxiv.org/pdf/1505.00853.pdf
* Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
https://arxiv.org/pdf/1502.03167.pdf
* Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
https://arxiv.org/pdf/1602.07868.pdf

