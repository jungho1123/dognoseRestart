# train.py 파일의 일부 또는 test_dataloader.py와 같은 파일로 만들어 실행해보세요.

import torch
from torch.utils.data import DataLoader
from src.test_dataloder import SiameseNoseDataset   # 우리가 만들 Dataset 클래스
from src.transforms import get_train_transform # 우리가 만들 transform 함수

# --- 설정 (config.yaml에서 불러올 값들) ---
TRAIN_CSV_PATH = 'data/pet_biometric_challenge_2022/train/train_data.csv'
TRAIN_IMG_DIR = 'data/pet_biometric_challenge_2022/train/images'
IMAGE_SIZE = 224
BATCH_SIZE = 16
# --- 설정 끝 ---

def test_data_loading():
    """
    Dataset과 DataLoader가 정상적으로 작동하는지 테스트하는 함수
    """
    print("="*30)
    print("데이터 로딩 테스트를 시작합니다...")
    
    # 1. Transform 정의
    train_transform = get_train_transform(IMAGE_SIZE, IMAGE_SIZE)

    # 2. Dataset 객체 생성
    try:
        train_dataset = SiameseNoseDataset(
            data_csv_path=TRAIN_CSV_PATH,
            image_dir=TRAIN_IMG_DIR,
            transform=train_transform
        )
        print("Dataset 객체 생성 성공!")
    except Exception as e:
        print(f"Dataset 객체 생성 중 오류 발생: {e} ❌")
        return

    # 3. DataLoader 객체 생성
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2 # Windows에서는 0 또는 2로 시작하는 것이 안정적일 수 있습니다.
    )
    print("DataLoader 객체 생성 성공!")

    # 4. 데이터 로딩 테스트 (첫 번째 배치 가져오기)
    try:
        # iter()로 DataLoader를 반복 가능한 객체로 만들고, next()로 첫 배치를 가져옵니다.
        img1, img2, labels = next(iter(train_loader))

        print("\n🎉 데이터 로딩 테스트 성공! ✅")
        # torch.Size([16, 3, 224, 224]) 와 같은 형태가 나와야 합니다.
        print(f"  - 첫 번째 이미지 배치(img1) 형태: {img1.shape}")
        # torch.Size([16, 3, 224, 224]) 와 같은 형태가 나와야 합니다.
        print(f"  - 두 번째 이미지 배치(img2) 형태: {img2.shape}")
        # torch.Size([16]) 와 같은 형태가 나와야 합니다.
        print(f"  - 레이블 배치(labels) 형태: {labels.shape}")
        # 0.0과 1.0이 섞여서 나와야 합니다. (Positive/Negative 페어)
        print(f"  - 레이블 예시: {labels[:10]}")
        print("="*30)

    except Exception as e:
        print(f"\nDataLoader에서 배치를 가져오는 중 오류 발생: {e} ❌")
        print("  - CSV 파일 경로, 이미지 경로, Dataset 코드의 __getitem__ 부분을 확인해보세요.")
        print("="*30)

if __name__ == '__main__':
    test_data_loading()