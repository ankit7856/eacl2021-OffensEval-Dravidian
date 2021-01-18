import os
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

for folder_name in ["epochs_3", "epochs_5", "epochs_20"]:
    select_dir = f"./arxiv-custom/semantic_similarity/bert-semantic-similarity/text_None/{folder_name}"
    loads = torch.load(os.path.join(select_dir, "model.pth.tar"),
                       map_location=torch.device(DEVICE))
    pytorch_model_bin = loads["model_state_dict"]
    torch.save(pytorch_model_bin, os.path.join(select_dir, "pytorch_model.bin"))
