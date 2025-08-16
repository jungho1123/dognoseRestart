# train.py 
import os
import yaml
import time
import torch
import torch.nn as nn # nn.functional F를 위해 필요할 수 있음
import torch.nn.functional as F # F.cosine_similarity 사용
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- 프로젝트 내 모듈 임포트 경로 확인 ---
# 아래 경로는 프로젝트 구조에 따라 상대 경로 또는 절대 경로로 수정 필요
# 예: from dataset.pair_dataset_contrastive import PairDataset
# 만약 train.py가 프로젝트 루트에 있고, dataset, model 등이 하위 폴더라면:
from dataset.pair_dataset_contrastive import PairDataset 
from model.siamese_cosine import SiameseNetwork         
from loss.contrastive_loss import ContrastiveLoss       
from utils.transforms import get_train_transform, get_val_transform 
# --- 임포트 경로 확인 완료 ---

from sklearn.metrics import f1_score, roc_auc_score

CONFIG_PATH = "config.yaml" 

def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_best_f1_threshold(preds, labels):
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        print("경고: 레이블이 비어있거나 단일 클래스만 포함하여 F1/AUC 계산이 불가합니다. 기본값을 반환합니다.")
        return 0.0, 0.5, 0.0 # F1, Thresh, AUC

    try:
        auc = roc_auc_score(labels, preds)
    except ValueError as e:
        print(f"AUC 계산 중 오류 발생: {e}. AUC를 0.0으로 설정합니다.")
        auc = 0.0

    # 임계값 범위는 예측값의 분포를 보고 좀 더 세밀하게 조정 가능
    min_pred, max_pred = preds.min(), preds.max()
    thresholds = np.linspace(min_pred, max_pred, 201) # 예측값 범위에 맞춰 임계값 설정
    
    best_f1 = 0.0
    best_thresh = 0.5 # 기본 임계값

    for t in thresholds:
        preds_bin = (preds >= t).astype(int) # 코사인 유사도는 높을수록 유사
        f1 = f1_score(labels, preds_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    return best_f1, best_thresh, auc

def main():
    cfg = load_config()
    
    # 시드 고정
    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(cfg["experiment"]["device"])
    print(f"사용 디바이스: {device}")
    print(f"현재 실험 설정된 백본: {cfg['model']['name']}")
    print(f"백본 입력 특징 수 (프로젝터 입력): {cfg['model']['in_features']}")


    # 모델 초기화
    model = SiameseNetwork(
        backbone_name=cfg["model"]["name"],
        in_features=cfg["model"]["in_features"], # run_backbone_batch.py에서 설정된 값 사용
        feature_dim=cfg["model"]["feature_dim"],
        pretrained=cfg["model"].get("pretrained", True)
    ).to(device)

    # 모델로부터 권장 입력 크기 가져오기 (C, H, W)
    if hasattr(model, 'recommended_input_size') and model.recommended_input_size:
        img_channels, img_height, img_width = model.recommended_input_size
        print(f"'{cfg['model']['name']}' 백본의 권장 입력 크기 ({img_channels}, {img_height}, {img_width})를 사용합니다.")
    else:
        # config.yaml의 image_size를 fallback으로 사용 (H, W) 또는 단일 값
        img_size_config = cfg["dataset"].get("image_size", 224) # 기본값 224
        if isinstance(img_size_config, int):
            img_height, img_width = img_size_config, img_size_config
        elif isinstance(img_size_config, (list, tuple)) and len(img_size_config) == 2:
            img_height, img_width = img_size_config[0], img_size_config[1]
        else: # 잘못된 형식일 경우 기본값 사용 및 경고
            print(f"경고: config의 dataset.image_size ({img_size_config}) 형식이 잘못되었습니다. 기본 224x224를 사용합니다.")
            img_height, img_width = 224, 224
        print(f"경고: 모델에서 권장 입력 크기를 찾을 수 없어, config의 image_size 또는 기본값 ({img_height}x{img_width})을 사용합니다.")


    # transforms.py의 함수 사용 (CLAHE 사용 여부는 config 또는 transforms.py 내부에서 결정 가능)
    use_clahe = cfg["dataset"].get("use_clahe_sharpen", True) # config에서 clahe 사용 여부 제어
    transform_train = get_train_transform(img_height, img_width, use_clahe_sharpen=use_clahe)
    transform_val = get_val_transform(img_height, img_width, use_clahe_sharpen=use_clahe)

    # 데이터셋 및 데이터로더
    image_root_path = Path(cfg["dataset"]["image_root"])
    train_csv_path = Path(cfg["dataset"]["train_csv"])
    val_csv_path = Path(cfg["dataset"]["val_csv"])

    # CSV 파일 경로가 image_root_path 기준으로 상대 경로일 수도, 절대 경로일 수도 있음.
    # PairDataset 내부에서 image_root와 합쳐지므로, csv 내 경로는 image_root 기준 상대경로여야 함.
    # config의 csv 경로는 image_root를 포함하지 않는 상대경로로 가정 (예: mini/train_pairs.csv)
    # 또는, Path(cfg["dataset"]["train_csv"])가 이미 data/mini/train_pairs.csv 형태의 절대/상대 경로일 수 있음
    # 여기서는 config의 경로가 이미 적절히 설정되어 있다고 가정.

    train_dataset = PairDataset(str(train_csv_path), image_root=str(image_root_path), transform1=transform_train, transform2=transform_train)
    val_dataset = PairDataset(str(val_csv_path), image_root=str(image_root_path), transform1=transform_val, transform2=transform_val)
    
    print(f"학습 데이터 CSV: {train_csv_path}, 검증 데이터 CSV: {val_csv_path}")
    print(f"데이터셋 이미지 루트: {image_root_path}")
    print(f"학습 데이터셋 크기: {len(train_dataset)}, 검증 데이터셋 크기: {len(val_dataset)}")


    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

    criterion = ContrastiveLoss(margin=cfg["train"]["margin"], reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    
    # 스케줄러 (예: CosineAnnealingLR)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"] * len(train_loader), eta_min=1e-7)


    best_val_metric = 0.0 # F1 또는 AUC를 기준으로 할 수 있음, 여기서는 F1 기준
    save_path_full = Path(cfg["save"]["save_dir"]) / cfg["save"]["save_name"]
    # save_dir은 run_backbone_batch.py에서 이미 생성됨

    log_lines_detail = [] 
    log_file_path_detail = Path(cfg["save"]["save_dir"]) / f"training_epoch_details_{cfg['model']['name']}.txt"

    start_time_total_train = time.time()
    print(f"\n학습 시작: 총 에폭 {cfg['train']['epochs']}, 배치 크기 {cfg['train']['batch_size']}")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_train_loss_sum = 0.0
        epoch_start_time = time.time()

        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Train]", unit="batch")

        for img1, img2, label, weight, _ in progress_bar_train:
            img1, img2 = img1.to(device), img2.to(device)
            label, weight = label.float().to(device), weight.float().to(device)

            optimizer.zero_grad()
            z1, z2 = model(img1, img2)
            
            loss_per_sample = criterion(z1, z2, label)
            weighted_loss = (loss_per_sample * weight).mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            # 스케줄러 사용 시 (매 스텝마다 업데이트하는 경우)
            # scheduler.step()

            epoch_train_loss_sum += weighted_loss.item() * img1.size(0) # 배치 손실의 합
            progress_bar_train.set_postfix(loss=weighted_loss.item(), lr=optimizer.param_groups[0]['lr'])

        avg_epoch_train_loss = epoch_train_loss_sum / len(train_loader.dataset)
        
        # 검증 단계
        model.eval()
        all_val_scores = []
        all_val_labels = []
        val_loss_sum = 0.0 # 검증 손실도 계산 (선택 사항)
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Val]", unit="batch")
        
        with torch.no_grad():
            for img1, img2, label, weight, _ in progress_bar_val:
                img1, img2 = img1.to(device), img2.to(device)
                val_label, val_weight = label.float().to(device), weight.float().to(device)
                
                z1, z2 = model(img1, img2) # 모델 forward는 이미 normalize된 z1, z2 반환
                
                # 검증 손실 계산
                val_loss_per_sample = criterion(z1, z2, val_label)
                val_weighted_loss = (val_loss_per_sample * val_weight).mean()
                val_loss_sum += val_weighted_loss.item() * img1.size(0)

                cos_sim = F.cosine_similarity(z1, z2, dim=-1) # 이미 normalize 되어 있으므로 (z1*z2).sum()과 동일
                
                all_val_scores.extend(cos_sim.cpu().numpy())
                all_val_labels.extend(val_label.cpu().numpy())
                progress_bar_val.set_postfix(loss=val_weighted_loss.item())

        avg_epoch_val_loss = val_loss_sum / len(val_loader.dataset)
        all_val_scores_np = np.array(all_val_scores)
        all_val_labels_np = np.array(all_val_labels)
        
        val_f1, best_threshold_epoch, val_auc = compute_best_f1_threshold(all_val_scores_np, all_val_labels_np)
        
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        log_output_epoch = (f"Epoch [{epoch+1}/{cfg['train']['epochs']}] - "
                            f"LR: {current_lr:.7f}, "
                            f"TrainLoss: {avg_epoch_train_loss:.4f}, ValLoss: {avg_epoch_val_loss:.4f}, "
                            f"ValF1: {val_f1:.4f}, ValAUC: {val_auc:.4f}, BestThresh: {best_threshold_epoch:.3f}, "
                            f"Time: {epoch_duration:.2f}s")
        print(log_output_epoch)
        log_lines_detail.append(log_output_epoch)

        # 스케줄러 사용 시 (매 에폭마다 업데이트하는 경우, 예: ReduceLROnPlateau)
        # scheduler.step(avg_epoch_val_loss) 또는 scheduler.step(val_f1)
        
        # 모델 저장 기준 (예: Val F1)
        if val_f1 > best_val_metric:
            best_val_metric = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_f1': best_val_metric,
                'best_threshold': best_threshold_epoch,
                'val_auc_at_best_f1': val_auc,
                'config': cfg # 현재 config도 함께 저장
            }, save_path_full)
            print(f"*** 최고 성능 모델 저장됨: {save_path_full} (F1: {val_f1:.4f}, Thresh: {best_threshold_epoch:.3f}) ***")

    total_training_duration_str = f"\n총 학습 시간: {(time.time() - start_time_total_train)/60:.2f} 분"
    print(total_training_duration_str)
    log_lines_detail.append(total_training_duration_str)
    
    with open(log_file_path_detail, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines_detail))
    print(f"상세 에폭별 학습 로그가 '{log_file_path_detail}'에 저장되었습니다.")

if __name__ == "__main__":
    main()