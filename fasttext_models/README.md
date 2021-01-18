```
cd Codemixed/
mkdir fasttext_models
cd fasttext_models
pip install fasttext
python3
>>> import fasttext.util
>>> fasttext.util.download_model('en', if_exists='ignore')
>>> fasttext.util.download_model('hi', if_exists='ignore')
>>> ft = fasttext.load_model('cc.en.300.bin')
```