
Steps to populate this repo with the required files:
```
git clone https://github.com/libindic/indic-trans
```

To run inline, need to run the following command
```
cd indic-trans
python setup.py build_ext --inplace
```

Then, move ```./indictrans``` as below
```
mv ./indictrans ../
cd ../
rm -rf ./indic-trans
```
