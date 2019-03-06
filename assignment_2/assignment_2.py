from sklearn.datasets import load_files

train_data = load_files('assignment_2/dataset/train-data', load_content = True, shuffle = True, random_state = 42)
print(train_data.target_names)