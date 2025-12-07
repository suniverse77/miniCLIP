import torch
import torch
import time
import numpy as np

def measure_inference_speed(model, args, device, batch_size=None, iterations=100, warmup=100):
    """
        iterations: 측정 반복 횟수 (평균을 내기 위함)
        warmup: 예열 횟수
    """
    model.eval()
    
    bs = batch_size if batch_size is not None else args.batch_size
    
    dummy_input = torch.randn(bs, 3, args.img_size, args.img_size).to(device)

    # 1. Warm-up (GPU 예열)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            
    # GPU 동기화 (이전 작업이 모두 끝날 때까지 대기)
    if device == 'cuda':
        torch.cuda.synchronize()

    # 2. 실제 측정 시작
    timings = []
    
    with torch.no_grad():
        for _ in range(iterations):
            # 시작 시간 기록
            if device == 'cuda':
                torch.cuda.synchronize() # 확실한 측정을 위해 시작 전 동기화
            start_time = time.time()
            
            # 모델 추론
            _ = model(dummy_input)
            
            # 종료 시간 기록
            if device == 'cuda':
                torch.cuda.synchronize() # GPU 연산이 끝날 때까지 대기
            end_time = time.time()
            
            timings.append(end_time - start_time)

    # 3. 결과 계산
    timings = np.array(timings)
    avg_latency = np.mean(timings) # 초 단위 평균
    
    # Latency: 배치 하나를 처리하는 데 걸리는 시간 (ms)
    latency_ms = avg_latency * 1000
    
    # Throughput: 1초당 처리할 수 있는 이미지 개수 (images/sec)
    # 계산식: (배치 크기) / (배치 하나 처리하는 평균 시간)
    throughput = bs / avg_latency
    
    print(f"  >> Average Latency        : {latency_ms:.4f} ms / batch")
    print(f"  >> Throughput             : {throughput:.2f} images / sec")
    print("-------------------------------------------------------------")

def calcul_params(model):
    total_params = 0
    learnable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()  # numel()이 파라미터의 총 개수를 반환
        if param.requires_grad:
            learnable_params += param.numel()
            
    print(f"  >> Frozen Parameters      : {(total_params - learnable_params):,}")
    print(f"  >> Learnable Parameters   : {learnable_params:,}")
    print(f"  >> Total Parameters       : {total_params:,}")
