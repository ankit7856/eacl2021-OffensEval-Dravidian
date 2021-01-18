import os

select_dir = "./sentimix2020/Hinglish/xlm-roberta-base"
dirs = os.listdir(select_dir)
for dir_ in dirs:
    model_file_name = os.path.join(select_dir, dir_, "model.pth.tar")
    os.system(f"rm {model_file_name}")
    print(f"removed {model_file_name}")

select_dir = "./sail2017/Hinglish/xlm-roberta-base"
dirs = os.listdir(select_dir)
for dir_ in dirs:
    model_file_name = os.path.join(select_dir, dir_, "model.pth.tar")
    os.system(f"rm {model_file_name}")
    print(f"removed {model_file_name}")