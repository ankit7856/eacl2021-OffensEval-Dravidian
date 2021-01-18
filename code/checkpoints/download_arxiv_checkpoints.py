#############################################
# USAGE
# -----
# python download_checkpoints.py
#############################################

# taken from https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python
# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039

import requests
import os


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def create_paths(path_: str):
    if not os.path.exists(path_):
        os.makedirs(path_)
        print(f"{path_} created")
    else:
        print(f"{path_} already exists")


if __name__ == "__main__":

    """ arxiv - sail2017 """
    base_path = "./arxiv-sail2017/Hinglish"
    # baseline
    save_path = os.path.join(base_path, "baseline/xlm-roberta-base/text_raw")
    create_paths(save_path)
    download_file_from_google_drive('1SXZ2uECF5hvazfHT5YXA80WyeuZo0wpd', os.path.join(save_path, "label_vocab.json"))
    download_file_from_google_drive('1gPtWsJSAvLPNDsUiieWkk6T7U1eHC3sZ', os.path.join(save_path, "model.pth.tar"))
    # baseline (trans)
    save_path = os.path.join(base_path, "baseline (trans)/xlm-roberta-base/text_raw")
    create_paths(save_path)
    download_file_from_google_drive('1ClzAdQ678P9wYb1Y63MXxDYlrUZKZPxN', os.path.join(save_path, "label_vocab.json"))
    download_file_from_google_drive('1Qmhh6RjRwjXuabhH5ysxRDIhZXZX3GnN', os.path.join(save_path, "model.pth.tar"))
    # data_aug with MLM pretraining
    save_path = os.path.join(base_path, "data_aug with MLM pretraining/xlm-roberta-base/text_raw")
    create_paths(save_path)
    download_file_from_google_drive('1PKiLUqB8HUMqtFSQFKUr_CJ_bEl1PpBj', os.path.join(save_path, "label_vocab.json"))
    download_file_from_google_drive('1kKUqPsXEeoX1e0Zvgv3mDT7K79syPONi', os.path.join(save_path, "model.pth.tar"))

    """ arxiv - sentimix2020 """
    base_path = "./arxiv-sentimix2020/Hinglish"
    # baseline
    save_path = os.path.join(base_path, "baseline/xlm-roberta-base/text_raw")
    create_paths(save_path)
    download_file_from_google_drive('1LOOnnmC_HF2MZ6l9NT-gtcf3f_p87dJX', os.path.join(save_path, "label_vocab.json"))
    download_file_from_google_drive('1_BttDwmFjZUa5h7U1271tWNyuC_5aH6E', os.path.join(save_path, "model.pth.tar"))
    # baseline (trans)
    save_path = os.path.join(base_path, "baseline (trans)/xlm-roberta-base/text_raw")
    create_paths(save_path)
    download_file_from_google_drive('1iQrkujvST0A8Bv3wAU3YmL9NZTFDxlLi', os.path.join(save_path, "label_vocab.json"))
    download_file_from_google_drive('1ZjF8WrGZAvR5UAVJW2sGGEdHYWGqunz-', os.path.join(save_path, "model.pth.tar"))
    # data_aug with MLM pretraining
    save_path = os.path.join(base_path, "data_aug with MLM pretraining/xlm-roberta-base/text_raw")
    create_paths(save_path)
    download_file_from_google_drive('18uF40MIOD_KixXusJP848xp8JLT1ijV-', os.path.join(save_path, "label_vocab.json"))
    download_file_from_google_drive('1mSqDUV1GnhSaVzqnyX1x1GazeFQJk7n6', os.path.join(save_path, "model.pth.tar"))

    print("complete")
