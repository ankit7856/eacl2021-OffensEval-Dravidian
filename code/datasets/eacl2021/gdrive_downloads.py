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

    root_save_path = "./offeval"

    save_path = os.path.join(root_save_path, "tamil")
    create_paths(save_path)
    download_file_from_google_drive('15auwrFAlq52JJ61u7eSfnhT9rZtI5sjk', os.path.join(save_path, "train.tsv"))
    download_file_from_google_drive('1Jme-Oftjm7OgfMNLKQs1mO_cnsQmznRI', os.path.join(save_path, "dev.tsv"))
    download_file_from_google_drive('17o-spkU5JnI_18qDJXO-F_DoIG9Zzv9q', os.path.join(save_path, "test.tsv"))
    download_file_from_google_drive('10RHrqXvIKMdnvN_tVJa_FAm41zaeC8WN', os.path.join(save_path, "tamil_offensive_full_test_with_labels.csv"))

    save_path = os.path.join(root_save_path, "malayalam")
    create_paths(save_path)
    download_file_from_google_drive('13JCCr-IjZK7uhbLXeufptr_AxvsKinVl', os.path.join(save_path, "train.tsv"))
    download_file_from_google_drive('1J0msLpLoM6gmXkjC6DFeQ8CG_rrLvjnM', os.path.join(save_path, "dev.tsv"))
    download_file_from_google_drive('1waRFe4yTG8TMkMruICaavd9JH0xiO_rb', os.path.join(save_path, "test.tsv"))
    download_file_from_google_drive('1zx1wCC9A-Pp80mzbqixb52WlWQQ7ATyJ', os.path.join(save_path, "mal_full_offensive_test_with_labels.csv"))

    save_path = os.path.join(root_save_path, "kannada")
    create_paths(save_path)
    download_file_from_google_drive('1BFYF05rx-DK9Eb5hgoIgd6EcB8zOI-zu', os.path.join(save_path, "train.tsv"))
    download_file_from_google_drive('1V077dMQvscqpUmcWTcFHqRa_vTy-bQ4H', os.path.join(save_path, "dev.tsv"))
    download_file_from_google_drive('14DQvnNZCXSgmiZxJqGtPYFdRqBH7TOSr', os.path.join(save_path, "test.tsv"))
    download_file_from_google_drive('1Px2CvIkLP_xaNhz_fCofW-7GGBCnSYsa', os.path.join(save_path, "kannada_offensive_test_with_labels.csv"))

    print("complete")
