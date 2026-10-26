# fetching VCTK+DEMAND from drive
!gdown --id 159Z6f7j0T-bSZx8z98Y9jK4MXnBX8xOR
!gdown --id 1M4RhDerJCS3R-BMIHCkfUX6j94EG0-0r
zip_path = '/kaggle/working/voice_demand.zip'
extract_dir = '/kaggle/working/' 
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)


    
train_json_path  = '/kaggle/working/voicebank_wav/train'
clean_train_path = '/kaggle/working/voicebank_wav/train/clean'
noisy_train_path = '/kaggle/working/voicebank_wav/train/noisy'

test_json_path  = '/kaggle/working/voicebank_wav/test'
clean_test_path = '/kaggle/working/voicebank_wav/test/clean'
noisy_test_path = '/kaggle/working/voicebank_wav/test/noisy'



def get_info(data_path, data_list):
    data_path_list = []
    for file in tqdm(data_list):
        file  = opj(data_path, file)
        infos = [file, torchaudio.load(file)[0].shape[-1]]
        data_path_list.append(infos)
    return data_path_list

def make_json(clean_path, noisy_path, json_path):
    clean_list = os.listdir(clean_path)
    noisy_list = os.listdir(noisy_path)
    clean_list.sort(); noisy_list.sort()
    clean_list = get_info(clean_path, clean_list)
    noisy_list = get_info(noisy_path, noisy_list)

    if not os.path.exists(json_path):
        os.makedirs(json_path)
    with open(opj(json_path, 'clean.json'), 'w') as file:
        json.dump(clean_list, file, indent=4)
    with open(opj(json_path, 'noisy.json'), 'w') as file:
        json.dump(noisy_list, file, indent=4)
    print('---json is generated---')

if __name__ == "__main__":
    print('--- Trainset Json ---')
    make_json(clean_train_path, noisy_train_path, train_json_path)
    print('--- Testset Json ---')
    make_json(clean_test_path, noisy_test_path, test_json_path)