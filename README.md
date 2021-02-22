# EACL 2021 OffensEval in Dravidian Languages

### Leaderboard (Team SJ_AJ)
| Language  | Rank | Precision | Recall | F1   |
|-----------|------|-----------|--------|------|
| Kannada   | 1    | 0.73      | 0.78   | 0.75 |
| Malayalam | 2    | 0.97      | 0.97   | 0.96 |
| Tamil     | 3    | 0.75      | 0.79   | 0.76 |
  
### Data & Models
- See [./code/datasets/eacl2021](./code/datasets/eacl2021) for creating data and running code. However, the main scripts are located in [code/scripts](.code/scripts) folder.
- The .ipynb files (colab files) use absolute paths. So be careful and please change them according to your local directory path(s).
- See [./code/datasets/eacl2021/run.sh](./code/datasets/eacl2021/run.sh) for how to run the code. Alternatively, see notebooks in [./colab_notebooks](./colab_notebooks)

### Pretrained Models
- Pretrained models, submission files and training checkpoints can be downloaded from this [drive repo](https://drive.google.com/drive/folders/1xnQ63uZ7Pq1go1K21OkgRkGFUkfizoT7?usp=sharing).
- Scripts for task-adaptive pretraining are placed at [./pretraining](./pretraining)

### Citation
```
@misc{jayanthi2021sjajdravidianlangtecheacl2021,
      title={SJ_AJ@DravidianLangTech-EACL2021: Task-Adaptive Pre-Training of Multilingual BERT models for Offensive Language Identification}, 
      author={Sai Muralidhar Jayanthi and Akshat Gupta},
      year={2021},
      eprint={2102.01051},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Contact
Feel free to contact us for a quick chat or a discussion at [gmail](jsaimurali001@gmail.com)