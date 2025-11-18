from calflops import calculate_flops

import torch

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # calflops는 이 forward 함수를 추적하게 됩니다.
        # 반환값은 연산량 계산에 영향을 주지 않으므로
        # 원래 함수를 그대로 호출하면 됩니다.
        return self.model.detect_forward_seg(x)

def calcul_flops(model, args):
    # 2. 모델이 현재 사용 중인 디바이스 확인 (e.g., 'cuda' or 'cpu')
    device = next(model.parameters()).device

    # 3. 래퍼 모듈로 모델을 감싸고, 동일한 디바이스로 보냅니다.
    wrapped_model = ModelWrapper(model).to(device)
    
    # 4. input_shape을 정의합니다.
    # (배치 크기 1, 채널 3, 이미지 크기, 이미지 크기)
    test_input_shape = (1, 3, args.img_size, args.img_size)

    # 5. calculate_flops에 래핑된 모델을 전달합니다.
    flops, macs, params = calculate_flops(
        model=wrapped_model,  # 래퍼 모델 사용
        input_shape=test_input_shape,  # input_shape 사용
        output_as_string=True, 
        output_precision=4
    ) 

    # 'ResNet'이라고 되어 있는 부분은 원하시면 'Model' 등으로 바꾸셔도 됩니다.
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

def calcul_params(model):
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()  # numel()이 파라미터의 총 개수를 반환
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"  >> 총 파라미터 (Total Parameters): {total_params:,}")
    print(f"  >> 학습 가능 파라미터 (Trainable): {trainable_params:,}")
    print(f"  >> 고정된 파라미터 (Frozen): {(total_params - trainable_params):,}")
