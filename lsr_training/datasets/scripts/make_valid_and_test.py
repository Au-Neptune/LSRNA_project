import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import pickle
import time

import sys
sys.path.append('../..')

base_dir = '/home/m11215122/datasets/OpenImages/' # fixed

annotation_path = f'{base_dir}/image_ids_and_rotation.csv' # metadata of OpenImages 
print('loading annotation file...')
urls = list(pd.read_csv(annotation_path)['OriginalURL'])

processed_info = {}
processed_info_path = f'{base_dir}/process_info_1_1.pkl'
if os.path.exists(processed_info_path):
    with open(processed_info_path, 'rb') as f:
        processed_info = pickle.load(f)

def get_image(url):
    global processed_info
    session = requests.Session()
    try:
        img_name = url.split('/')[-1].split('?')[0]
        if img_name[-4:].lower() not in ['.jpg', 'jpeg']:
            return None, None
        assert img_name.count('.') == 1
        img_name = img_name.split('.')[0] # w/o extension

        key = f'{base_dir}/HR/{img_name}_s000.jpg'
        if key in processed_info:
            print(f'[skip] files already exists for {img_name}')
            return None, None

        response = session.get(url, timeout=2)
        response.raise_for_status() 
        img = Image.open(BytesIO(response.content))

        width, height = img.size
        if height >= 3072 and width >= 3072 and img.mode == 'RGB':
            return img, img_name
        return None, None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None
    except Exception as e:
        print(f"Other error occurred: {e}")
        return None, None
    finally:
        session.close()

save_dir_valid = os.path.join(base_dir, 'valid')
save_dir_test = os.path.join(base_dir, 'test')
os.makedirs(save_dir_valid, exist_ok=True)
os.makedirs(save_dir_test, exist_ok=True)

count = 0
for url in urls[::-1]:
    time.sleep(1)  # to avoid overwhelming the server
    img, base_name = get_image(url)
    if img is None: continue

    # 超過 3k 直接存做 valid 和 test
    if count < 450:
        img.save(os.path.join(save_dir_valid, f'{base_name}.jpg'))
        print(f'[valid] saved {base_name}.jpg')
    elif count < 1500:
        img.save(os.path.join(save_dir_test, f'{base_name}.jpg'))
        print(f'[test] saved {base_name}.jpg')
    else:
        break

    count += 1