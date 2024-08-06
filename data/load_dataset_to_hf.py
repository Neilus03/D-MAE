from datasets import load_dataset


dataset = load_dataset("imagefolder", data_dir="/home/ndelafuente/Desktop/D-MAE/data/data")

dataset.push_to_hub("neildlf/depth_coco", commit_message="load dataset to hub")
