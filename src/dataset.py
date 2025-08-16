# pair_dataset.py (가칭)

import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SiameseNoseDataset(Dataset):
    def __init__(self, data_csv_path, image_dir, transform=None):
        """
        데이터셋 초기화 함수
        
        Args:
            data_csv_path (str): train_data.csv 또는 valid_data.csv 파일 경로
            image_dir (str): 이미지 파일들이 저장된 디렉토리 경로
            transform (callable, optional): 이미지에 적용할 transform.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # 1. CSV 파일을 읽어와서 DataFrame으로 만듭니다.
        self.df = pd.read_csv(data_csv_path)
        
        # 2. 'dog_id'를 기준으로 이미지 파일명들을 그룹화하여 딕셔너리로 저장합니다.
        #    이 구조가 클래스 불균형을 해결하는 핵심입니다!
        #    예: {'dog_01': ['img1.png', 'img2.png'], 'dog_02': [...]}
        self.dog_id_to_images = self.df.groupby('dog_id')['image_id'].apply(list).to_dict()
        
        # 3. 모든 dog_id 리스트를 저장해 둡니다. (샘플링에 사용)
        self.dog_ids = list(self.dog_id_to_images.keys())
        
        # 4. 이미지 수가 2개 이상인 dog_id만 따로 저장합니다. (Positive 페어 생성용)
        self.positive_dog_ids = [dog_id for dog_id, images in self.dog_id_to_images.items() if len(images) >= 2]

        print(f"총 {len(self.df)}개의 이미지 로드 완료.")
        print(f"총 {len(self.dog_ids)}마리의 강아지 ID 로드 완료.")


    def __len__(self):
        # 데이터셋의 전체 길이를 결정합니다. 
        # 보통 (전체 이미지 수 * 클래스 수) 정도로 충분히 크게 설정하여 다양한 페어가 생성되게 합니다.
        return len(self.df) * 4 

    def __getitem__(self, index):
        # 1:1 비율로 Positive / Negative 페어를 생성합니다.
        if index % 2 == 0:
            # 짝수 인덱스: Positive 페어 생성 (같은 강아지, 다른 이미지 2장)
            label = 1.0 # Positive
            
            # 1. Positive 페어를 만들 수 있는 강아지 ID 중에서 하나를 무작위로 선택
            selected_dog_id = random.choice(self.positive_dog_ids)
            
            # 2. 해당 강아지의 이미지 리스트에서 2개의 이미지를 무작위로 선택 (중복 없이)
            img_name1, img_name2 = random.sample(self.dog_id_to_images[selected_dog_id], 2)

        else:
            # 홀수 인덱스: Negative 페어 생성 (다른 강아지, 이미지 2장)
            label = 0.0 # Negative
            
            # 1. 서로 다른 강아지 ID 2개를 무작위로 선택
            dog_id1, dog_id2 = random.sample(self.dog_ids, 2)
            
            # 2. 각 강아지 ID에서 이미지를 하나씩 무작위로 선택
            img_name1 = random.choice(self.dog_id_to_images[dog_id1])
            img_name2 = random.choice(self.dog_id_to_images[dog_id2])

        # 이미지 경로 조합 및 이미지 로드
        img1_path = os.path.join(self.image_dir, img_name1)
        img2_path = os.path.join(self.image_dir, img_name2)
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # Transform 적용
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)
    
# src/dataset.py 파일에 이어서 추가합니다.

# ... 기존 SiameseNoseDataset 클래스 코드 ...

class ValidationPairDataset(Dataset):
    def __init__(self, data_csv_path, image_dir, transform=None):
        """
        검증 데이터셋 초기화 함수
        - valid_data.csv 파일은 'image_id1', 'image_id2', 'label' 컬럼을 가지고 있다고 가정합니다.
        
        Args:
            data_csv_path (str): valid_data.csv 파일 경로
            image_dir (str): 검증 이미지 파일들이 저장된 디렉토리 경로
            transform (callable, optional): 이미지에 적용할 transform.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # 1. 검증용 CSV 파일을 읽어옵니다.
        self.df = pd.read_csv(data_csv_path)
        
        print(f"검증용 페어 {len(self.df)}개 로드 완료.")

    def __len__(self):
        # 전체 길이는 CSV 파일의 행의 개수와 같습니다.
        return len(self.df)

    def __getitem__(self, index):
        # 1. index에 해당하는 행(row)을 가져옵니다.
        row = self.df.iloc[index]
        
        # 2. 두 이미지 파일명과 레이블을 가져옵니다.
        img_name1 = row['image_id1']
        img_name2 = row['image_id2']
        label = row['label']

        # 3. 이미지 경로를 조합하고 이미지를 로드합니다.
        img1_path = os.path.join(self.image_dir, img_name1)
        img2_path = os.path.join(self.image_dir, img_name2)
        
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다. 경로: {img1_path} 또는 {img2_path}")
            # 오류 발생 시 빈 텐서를 반환하거나 다른 방식으로 처리할 수 있습니다.
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), torch.tensor(-1.0)

        # 4. Transform을 적용합니다. (데이터 증강이 없는 get_val_transform 사용)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)