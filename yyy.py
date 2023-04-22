import os
import shutil


os.makedirs("./model_weights_archive", exist_ok=True)
target_files = ["info.pkl", "test_log.txt", "train_log.txt"]
for exp_name in sorted(os.listdir("./model_weights")):
    for target_file in target_files:
        os.makedirs(f"./model_weights_archive/{exp_name}", exist_ok=True)
        try:
            shutil.copyfile(
                src=f"./model_weights/{exp_name}/{target_file}",
                dst=f"./model_weights_archive/{exp_name}/{target_file}")
        except FileNotFoundError:
            print(f"[!] file {exp_name}/{target_file} not found.")
