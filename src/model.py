# model/siamese_cosine.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.backbone_build import get_backbone # 실제 경로 확인

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_name: str, in_features: int, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        
        # get_backbone은 이제 (model, input_size_tuple)을 반환합니다.
        backbone_model = get_backbone(backbone_name, pretrained=pretrained)
        self.backbone = backbone_model
         # (C, H, W) 형태, train.py에서 사용

        # 프로젝터의 입력은 백본의 출력 특징 차원(in_features)을 사용합니다.
        # 이 in_features 값은 run_backbone_batch.py에 의해 config.yaml에 설정되고,
        # train.py에서 읽어와 SiameseNetwork 생성 시 전달됩니다.
        
        # 프로젝터 첫 번째 은닉층 크기 동적 설정 (예시 - 이전 답변 참고)
        proj_hidden1 = 1024 
        if in_features > 1024:
            # 예시: in_features를 유지하거나, 적절히 줄이거나, 고정값 사용
            proj_hidden1 = min(in_features, 2048) # 예: 최대 2048, 그보다 작으면 in_features
            # proj_hidden1 = in_features # 백본 출력 차원 유지 옵션
        elif in_features < 512:
             proj_hidden1 = 512 

        proj_hidden2 = 512
        if proj_hidden1 <= proj_hidden2 : 
            proj_hidden2 = max(feature_dim, proj_hidden1 // 2)
        
        print(f"프로젝터 구성: 입력={in_features} -> 은닉1={proj_hidden1} -> 은닉2={proj_hidden2} -> 출력={feature_dim}")

        # 프로젝터 구성 단순화 또는 계층 수 조정 가능
        if proj_hidden1 == proj_hidden2 and proj_hidden1 == feature_dim: # 입력 -> 출력
             self.projector = nn.Linear(in_features, feature_dim)
        elif proj_hidden1 == proj_hidden2: # 입력 -> 은닉1 -> 출력
            self.projector = nn.Sequential(
                nn.Linear(in_features, proj_hidden1),
                nn.BatchNorm1d(proj_hidden1),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(proj_hidden1, feature_dim)
            )
        else: # 입력 -> 은닉1 -> 은닉2 -> 출력 (기존과 유사)
            self.projector = nn.Sequential(
                nn.Linear(in_features, proj_hidden1),
                nn.BatchNorm1d(proj_hidden1),
                nn.GELU(),
                nn.Dropout(0.2),

                nn.Linear(proj_hidden1, proj_hidden2),
                nn.BatchNorm1d(proj_hidden2),
                nn.GELU(),
                nn.Dropout(0.2),

                nn.Linear(proj_hidden2, feature_dim)
            )

    def extract(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.backbone(x)
        # timm 모델 중 num_classes=0으로 로드 시, global_pool까지 적용된 (B, Features) 형태를 반환.
        # 만약 백본 출력이 (B, C, H, W) 형태라면 여기서 AdaptiveAvgPool2d 등이 필요할 수 있으나,
        # 현재 get_backbone 설정(num_classes=0, global_pool='avg')은 이미 특징 벡터를 반환.
        x = self.projector(x)
        return F.normalize(x, dim=1) if normalize else x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = self.extract(x1, normalize=True)
        z2 = self.extract(x2, normalize=True)
        return z1, z2