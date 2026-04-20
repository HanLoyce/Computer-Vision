import numpy as np
import os 
import gzip
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_URLS = {
    'train_images': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
    'train_labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
    'test_images': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
}

FILENAME_TO_KEY = {
    'train-images-idx3-ubyte.gz': 'train_images',
    'train-labels-idx1-ubyte.gz': 'train_labels',
    't10k-images-idx3-ubyte.gz': 'test_images',
    't10k-labels-idx1-ubyte.gz': 'test_labels',
}

# ---------------加载数据-------------------
def load_data(data_dir='data'):
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(SCRIPT_DIR, data_dir)
    os.makedirs(data_dir, exist_ok=True)

    def ensure_dataset_file(filename):
        gz_filename = filename + '.gz'
        if os.path.exists(filename) or os.path.exists(gz_filename):
            return

        gz_basename = os.path.basename(gz_filename)
        if gz_basename not in FILENAME_TO_KEY:
            raise FileNotFoundError(f'No download URL configured for: {gz_basename}')

        url_key = FILENAME_TO_KEY[gz_basename]
        url = DATASET_URLS[url_key]
        print(f'Downloading {gz_basename} from {url} ...')
        urllib.request.urlretrieve(url, gz_filename)
        print(f'Saved to {gz_filename}')

    def read_images(filename):
        ensure_dataset_file(filename)
        if os.path.exists(filename + '.gz'):
            with gzip.open(filename + '.gz', 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
        else:
            with open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28*28) / 255.0  # 归一化到 0~1 之间
    
    def read_labels(filename):
        ensure_dataset_file(filename)
        if os.path.exists(filename + '.gz'):
            with gzip.open(filename + '.gz', 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
        else:
            with open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    X_train_full = read_images(os.path.join(data_dir,'train-images-idx3-ubyte'))
    y_train_full = read_labels(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    X_test = read_images(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    y_test = read_labels(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))

    X_train,y_train = X_train_full[:-10000],y_train_full[:-10000]
    X_val,y_val = X_train_full[-10000:],y_train_full[-10000:]

    return X_train,y_train,X_val,y_val,X_test,y_test