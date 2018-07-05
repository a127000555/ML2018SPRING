README
===

### Prerequisite

* gensim 3.4.0
* tensorflow 1.9.0-rc1
* keras 2.2.0
* pyTorch 0.3.0
* pandas 0.22.0
* scipy 1.0.0
* jieba 0.39

### How to run

```bash
# 把所有 data 放到同個資料夾下

# 斷詞、word2vec（reproduce 請跳過這步）
python data.py

# training（reproduce 請跳過這步）
python train4.py
python train5.py
python train6.py
python train7.py

# 生成內積值
python score4.py
python score5.py
python score6.py
python score7.py

# 生成 csv 檔，供底下 voting 使用
python blend3.py > outb3.csv
python blend467.py > outb467.csv
```
* `train5.py` `score5.py`  用到的 `W.pt` `W_test.pt` 取自 [fasttext 的 pretrained vectors](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)，因爲原檔案太大，只附上有用到的部分。



* ```cd in "simple bow"```.
```bash
python3 similarity.py
'''
將training data放入trainig_data資料夾中，
把testing_data放入同個目錄裡面，將會產生"word_emb"
這個是gensim的pretrain model
'''
python3 prepro.py
'''
會產生出"train"和"corpus"，分別是training的句子對應，以及編號對應句子的dict。
'''
python3 train.py
'''
會產生出simple.hdf5，這是model。
'''
python3 predict.py
'''
會產生出"sim3.csv"，這是這個model所預測的選項
'''
python3 voting.py
'''
將前一個的作法所產生的兩種不同的model產出的csv檔，和"sim3.csv"寫在voting.py的第一行，
執行的時候會產生出"ans3.csv"，這個是最後的結果cat model_pieces* > "simple.hdf5"

'''
```
* Use trained model: please cd in this directory and `cat model_pieces* > "simple.hdf5"`
