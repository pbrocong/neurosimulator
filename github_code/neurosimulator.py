import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
from scipy.optimize import curve_fit
from sklearn.metrics import confusion_matrix
import seaborn as sns

# --- 1. 데이터 로딩  ---
def read_split_by_write_vh(file_path):
    if not os.path.exists(file_path):
        print(f"[에러] 파일을 찾을 수 없습니다: {file_path}")
        return None, None
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print("엑셀 파일에서 읽어온 열 이름:", df.columns.tolist())
        required_cols = ['Conductance', 'WriteVH', 'PulseNum']
        if not all(col in df.columns for col in required_cols):
            print(f"[에러] 필수 열 {required_cols} 중 일부가 없습니다.")
            return None, None
        ltp_df = df[df['WriteVH'] > 0].sort_values(by='PulseNum').reset_index(drop=True)
        ltd_df = df[df['WriteVH'] < 0].sort_values(by='PulseNum').reset_index(drop=True)
        if ltp_df.empty or ltd_df.empty:
            print("[에러] LTP 또는 LTD로 분류된 데이터가 없습니다.")
            return None, None
        print(f"LTP 데이터 수: {len(ltp_df)}, LTD 데이터 수: {len(ltd_df)}")
        return ltp_df, ltd_df
    except Exception as e:
        print(f"[에러] 파일을 읽는 중 문제가 발생했습니다: {e}")
        return None, None

# --- 2. NeuroSim 모델 피팅 클래스 ---
class NeuroSimFitter:
    def __init__(self, ltp_df, ltd_df, target_range=(-1.0, 1.0), ltp_fit_ratio=0.8, ltd_fit_ratio=0.8):
        """
        NeuroSim 모델 피팅 클래스.
        
        Args:
            ltp_df (pd.DataFrame): LTP 데이터.
            ltd_df (pd.DataFrame): LTD 데이터.
            target_range (tuple): 스케일링할 목표 가중치 범위.
            ltp_fit_ratio (float): LTP 피팅에 사용할 데이터의 상위 비율 (0.0 ~ 1.0). 
                                   초기 불안정 데이터를 제외하기 위함.
            ltd_fit_ratio (float): LTD 피팅에 사용할 데이터의 상위 비율.
        """
        print("\n[INFO] NeuroSim 모델 피팅 시작 (안정화 구간 최적화 적용)...")

        # --- 원본 데이터 전체를 저장 ---
        self.ltp_p_full = ltp_df['PulseNum'].values
        self.ltd_p_full = ltd_df['PulseNum'].values
        ltp_g_full_real = ltp_df['Conductance'].values
        ltd_g_full_real = ltd_df['Conductance'].values

        # --- 실제 컨덕턴스 범위는 전체 데이터를 기준으로 설정 ---
        self.g_min_real = min(ltp_g_full_real.min(), ltd_g_full_real.min())
        self.g_max_real = max(ltp_g_full_real.max(), ltd_g_full_real.max())
        self.target_min, self.target_max = target_range
        
        print(f"전체 실제 컨덕턴스 범위: [{self.g_min_real:.2e}, {self.g_max_real:.2e}]")
        print(f"목표 스케일링 범위: [{self.target_min}, {self.target_max}]")

        # ---  피팅에 사용할 안정화된 데이터 구간 선택 ---
        ltp_stable_start_index = int(len(ltp_df) * (1 - ltp_fit_ratio))
        ltd_stable_start_index = int(len(ltd_df) * (1 - ltd_fit_ratio))

        # 피팅에 사용할 데이터 (안정화 구간)
        self.ltp_p_fit = ltp_df['PulseNum'].values[ltp_stable_start_index:]
        ltp_g_fit_real = ltp_df['Conductance'].values[ltp_stable_start_index:]
        self.ltd_p_fit = ltd_df['PulseNum'].values[ltd_stable_start_index:]
        ltd_g_fit_real = ltd_df['Conductance'].values[ltd_stable_start_index:]
        
        print(f"LTP 피팅 데이터: {len(self.ltp_p_fit)}개 사용 (상위 {ltp_fit_ratio*100:.0f}%)")
        print(f"LTD 피팅 데이터: {len(self.ltd_p_fit)}개 사용 (상위 {ltd_fit_ratio*100:.0f}%)")

        # --- 스케일링은 피팅용 데이터에만 적용 ---
        self.ltp_g_scaled = self.scale(ltp_g_fit_real)
        self.ltd_g_scaled = self.scale(ltd_g_fit_real)

        # 피팅 시작점(G_min/G_max)을 안정화된 데이터의 첫 값으로 설정
        self.g_min_fit_scaled = self.scale(ltp_g_fit_real[0]) if len(ltp_g_fit_real) > 0 else self.target_min
        self.g_max_fit_scaled = self.scale(ltd_g_fit_real[0]) if len(ltd_g_fit_real) > 0 else self.target_max
        
        # --- 피팅 수행 ---
        self.A_LTP, self.B_LTP = self._fit_ltp()
        self.A_LTD, self.B_LTD = self._fit_ltd()

        if self.A_LTP is None or self.A_LTD is None:
            raise RuntimeError("NeuroSim 모델 피팅 실패.")

        print(f"[SUCCESS] 스케일링된 모델 피팅 완료:")
        print(f"  LTP params (scaled): A={self.A_LTP:.4f}, B={self.B_LTP:.4f}")
        print(f"  LTD params (scaled): A={self.A_LTD:.4f}, B={self.B_LTD:.4f}")
        
        # 플로팅 시 전체 데이터를 표시하되, 피팅 곡선은 새로 계산된 모델을 따름
        self._plot_fit(ltp_g_full_real, ltd_g_full_real)

    def scale(self, g_real):
        # 분모가 0이 되는 것을 방지
        if (self.g_max_real - self.g_min_real) == 0:
            return np.zeros_like(g_real) + self.target_min
        return ((g_real - self.g_min_real) / (self.g_max_real - self.g_min_real)) * \
               (self.target_max - self.target_min) + self.target_min

    def unscale(self, g_scaled):
        # 분모가 0이 되는 것을 방지
        if (self.target_max - self.target_min) == 0:
             return np.zeros_like(g_scaled) + self.g_min_real
        return ((g_scaled - self.target_min) / (self.target_max - self.target_min)) * \
               (self.g_max_real - self.g_min_real) + self.g_min_real

    def _fit_ltp(self):
        # 피팅할 데이터가 충분히 있는지 확인
        if len(self.ltp_p_fit) < 2:
            print("[WARNING] LTP 피팅을 위한 데이터가 부족합니다.")
            return 1.0, 1.0 # 기본값 반환
        def model(P, A, B): return B * (1 - np.exp(-(P - self.ltp_p_fit[0]) / A)) + self.g_min_fit_scaled
        try:
            p0 = [self.ltp_p_fit.mean(), self.target_max - self.g_min_fit_scaled]
            params, _ = curve_fit(model, self.ltp_p_fit, self.ltp_g_scaled, p0=p0, maxfev=10000)
            return params
        except (RuntimeError, ValueError) as e:
            print(f"[ERROR] LTP 데이터 피팅 실패: {e}")
            return None, None

    def _fit_ltd(self):
        if len(self.ltd_p_fit) < 2:
            print("[WARNING] LTD 피팅을 위한 데이터가 부족합니다.")
            return 1.0, 1.0 # 기본값 반환
        def model(P, A, B): return -B * (1 - np.exp(-(P - self.ltd_p_fit[0]) / A)) + self.g_max_fit_scaled
        try:
            p0 = [self.ltd_p_fit.mean(), self.g_max_fit_scaled - self.target_min]
            params, _ = curve_fit(model, self.ltd_p_fit, self.ltd_g_scaled, p0=p0, maxfev=10000)
            return params
        except (RuntimeError, ValueError) as e:
            print(f"[ERROR] LTD 데이터 피팅 실패: {e}")
            return None, None

    def _plot_fit(self, ltp_g_full_real, ltd_g_full_real):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.ltp_p_full, ltp_g_full_real, facecolors='none', edgecolors='b', label='LTP Data (Real)')
        plt.scatter(self.ltd_p_full, ltd_g_full_real, facecolors='none', edgecolors='r', label='LTD Data (Real)')
        
        ltp_fit_scaled = self.g_of_p_ltp(torch.from_numpy(self.ltp_p_full))
        ltd_fit_scaled = self.g_of_p_ltd(torch.from_numpy(self.ltd_p_full))
        
        plt.plot(self.ltp_p_full, self.unscale(ltp_fit_scaled.numpy()), 'b-', label='LTP Fit (Unscaled)')
        plt.plot(self.ltd_p_full, self.unscale(ltd_fit_scaled.numpy()), 'r-', label='LTD Fit (Unscaled)')
        
        plt.title('NeuroSim Model Fitting Result (on Real Scale)'); plt.xlabel('Pulse Number'); plt.ylabel('Conductance')
        plt.legend(); plt.grid(True); plt.show()
    
    def g_of_p_ltp(self, P): 
        p_start = self.ltp_p_fit[0] if len(self.ltp_p_fit) > 0 else 0
        effective_P = (P - p_start).clamp(min=0)
        return self.B_LTP * (1 - torch.exp(-effective_P / (self.A_LTP + 1e-9))) + self.g_min_fit_scaled

    def p_of_g_ltp(self, G_scaled): 
        eps = 1e-9
        p_start = self.ltp_p_fit[0] if len(self.ltp_p_fit) > 0 else 0
        arg = 1 - (G_scaled - self.g_min_fit_scaled) / (self.B_LTP + eps)
        delta_P = -self.A_LTP * torch.log(arg.clamp(min=eps))
        return delta_P + p_start

    def g_of_p_ltd(self, P): 
        p_start = self.ltd_p_fit[0] if len(self.ltd_p_fit) > 0 else 0
        effective_P = (P - p_start).clamp(min=0)
        return -self.B_LTD * (1 - torch.exp(-effective_P / (self.A_LTD + 1e-9))) + self.g_max_fit_scaled

    def p_of_g_ltd(self, G_scaled): 
        eps = 1e-9
        p_start = self.ltd_p_fit[0] if len(self.ltd_p_fit) > 0 else 0
        arg = 1 - (self.g_max_fit_scaled - G_scaled) / (self.B_LTD + eps)
        delta_P = -self.A_LTD * torch.log(arg.clamp(min=eps))
        return delta_P + p_start

# --- 3. 옵티마이저 ---
class NeuroSimOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-3, fitter=None, pulse_scaling_factor=1.0):
        if fitter is None: raise ValueError("Fitter 객체는 필수입니다.")
        self.fitter = fitter
        defaults = dict(lr=lr, pulse_scaling_factor=pulse_scaling_factor)
        super(NeuroSimOptimizer, self).__init__(params, defaults)
        self.pulse_states_ltp = {}
        self.pulse_states_ltd = {}
        for group in self.param_groups:
            for p in group['params']:
                param_id = id(p)
                g_scaled_clamped = p.data.clone().clamp(fitter.target_min, fitter.target_max)
                self.pulse_states_ltp[param_id] = fitter.p_of_g_ltp(g_scaled_clamped).nan_to_num(0.0)
                self.pulse_states_ltd[param_id] = fitter.p_of_g_ltd(g_scaled_clamped).nan_to_num(0.0)

    @torch.no_grad()
    def step(self, closure=None):
        fitter = self.fitter
        for group in self.param_groups:
            lr, psf = group['lr'], group['pulse_scaling_factor']
            for p in group['params']:
                if p.grad is None: continue
                param_id = id(p)
                grad, G_current_scaled = p.grad.data, p.data
                delta_W_ideal = -lr * grad
                G_target_scaled = (G_current_scaled + delta_W_ideal).clamp(fitter.target_min, fitter.target_max)
                ltp_mask, ltd_mask = grad < 0, grad > 0
                P_current_ltp, P_current_ltd = self.pulse_states_ltp[param_id], self.pulse_states_ltd[param_id]
                G_final_scaled = G_current_scaled.clone()
                if ltp_mask.any():
                    P_target_ltp = fitter.p_of_g_ltp(G_target_scaled[ltp_mask])
                    delta_P = (P_target_ltp - P_current_ltp[ltp_mask]).nan_to_num(0.0)
                    delta_P_scaled = delta_P * psf
                    delta_P_actual = torch.floor(delta_P_scaled + torch.rand_like(delta_P_scaled))
                    P_new_ltp = P_current_ltp[ltp_mask] + delta_P_actual
                    G_final_scaled[ltp_mask] = fitter.g_of_p_ltp(P_new_ltp)
                if ltd_mask.any():
                    P_target_ltd = fitter.p_of_g_ltd(G_target_scaled[ltd_mask])
                    delta_P = (P_target_ltd - P_current_ltd[ltd_mask]).nan_to_num(0.0)
                    delta_P_scaled = delta_P * psf
                    delta_P_actual = torch.floor(delta_P_scaled + torch.rand_like(delta_P_scaled))
                    P_new_ltd = P_current_ltd[ltd_mask] + delta_P_actual
                    G_final_scaled[ltd_mask] = fitter.g_of_p_ltd(P_new_ltd)
                p.data.copy_(G_final_scaled.clamp(fitter.target_min, fitter.target_max))
                g_scaled_clamped = p.data.clone()
                self.pulse_states_ltp[param_id] = fitter.p_of_g_ltp(g_scaled_clamped).nan_to_num(0.0)
                self.pulse_states_ltd[param_id] = fitter.p_of_g_ltd(g_scaled_clamped).nan_to_num(0.0)

# --- 4. 모델, 학습/테스트 루프 ---
def get_mnist_loaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class SimpleNet(nn.Module):
    def __init__(self, weight_min, weight_max):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        
        print(f"모델 가중치 초기화 시작... 목표 범위: [{weight_min:.2f}, {weight_max:.2f}]")
        for layer in [self.fc1, self.fc2]:
            nn.init.uniform_(layer.weight, a=weight_min, b=weight_max)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        print("모델 가중치 초기화 완료.")

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss, correct = 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / len(train_loader.dataset)
    print(f'Train Epoch: {epoch}\tLoss: {avg_loss:.6f}\tAccuracy: {correct}/{len(train_loader.dataset)} ({train_acc:.2f}%)')
    return train_acc

def test(model, device, test_loader, return_preds=False):
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if return_preds:
                all_preds.extend(pred.view(-1).cpu().numpy())
                all_targets.extend(target.view(-1).cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)')
    if return_preds:
        return test_acc, all_preds, all_targets
    else:
        return test_acc

# --- 5. 시각화 및 유틸리티 함수 ---
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Predicted vs. Actual)', fontsize=15)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_accuracy(train_hist, test_hist):
    plt.figure(figsize=(10, 5))
    plt.plot(train_hist, label='Train Accuracy', marker='o')
    plt.plot(test_hist, label='Test Accuracy', marker='o')
    plt.title('Epoch-wise Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.legend(); plt.grid(True); plt.ylim(0, 101); plt.xticks(range(len(train_hist)), [str(i+1) for i in range(len(train_hist))]); plt.show()

def plot_weight_distribution(model, fitter, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Weight Distribution {title_suffix} (Unscaled Conductance)', fontsize=16)
    layers_to_plot = {'fc1': model.fc1, 'fc2': model.fc2}
    for i, (name, layer) in enumerate(layers_to_plot.items()):
        weights_scaled = layer.weight.data.cpu().numpy().flatten()
        weights_unscaled = fitter.unscale(weights_scaled)
        ax = axes[i]
        ax.hist(weights_unscaled, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'{name} Weights')
        ax.set_xlabel('Conductance (S)')
        ax.set_ylabel('Density')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(fitter.g_min_real, fitter.g_max_real) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def save_accuracy_to_excel(train_acc_history, test_acc_history, filename="mnist_accuracy_log.xlsx"):
    try:
        df = pd.DataFrame({
            'Epoch': list(range(1, len(train_acc_history) + 1)),
            'Test Accuracy (%)': test_acc_history
        })
        df.to_excel(filename, index=False)
        print(f"[INFO] 에포크별 정확도 기록이 '{filename}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"[WARNING] 정확도 엑셀 저장 중 오류 발생: {e}")

# --- 6. 메인 실행 흐름 ---
if __name__ == "__main__":
    file_path = input("LTP/LTD 데이터가 포함된 엑셀 파일 경로를 입력하세요: ").strip().strip("'").strip('"')
    
    ltp_df, ltd_df = read_split_by_write_vh(file_path)

    if ltp_df is not None and ltd_df is not None:
        try:
            # --- 하이퍼파라미터 실험 영역  ---
            TARGET_RANGE = (-1.0, 1.0)
            BATCH_SIZE = 128      
            EPOCHS = 10  
            LEARNING_RATE = 0.0001  
            PULSE_SCALING_FACTOR = 300
            
            #  [피팅 개선] 피팅에 사용할 데이터 비율 (0.0 ~ 1.0)
            # 예를 들어 0.9는 데이터의 마지막 90% (안정된 구간)를 사용합니다.
            LTP_FIT_RATIO = 1.0
            LTD_FIT_RATIO = 1.0
            # -----------------------------------------------------------

            fitter = NeuroSimFitter(ltp_df, ltd_df, 
                                    target_range=TARGET_RANGE,
                                    ltp_fit_ratio=LTP_FIT_RATIO,
                                    ltd_fit_ratio=LTD_FIT_RATIO)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"\nUsing device: {device}")
            
            train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)
            
            model = SimpleNet(weight_min=TARGET_RANGE[0], weight_max=TARGET_RANGE[1]).to(device)
            
            plot_weight_distribution(model, fitter, title_suffix="Before Training")

            optimizer = NeuroSimOptimizer(model.parameters(), lr=LEARNING_RATE, fitter=fitter, pulse_scaling_factor=PULSE_SCALING_FACTOR)
            
            train_acc_history, test_acc_history = [], []
            print(f"\n[INFO] MNIST 데이터셋으로 NeuroSim 방식 학습 시작 (Epochs: {EPOCHS}, LR: {LEARNING_RATE}, PSF: {PULSE_SCALING_FACTOR}, Batch: {BATCH_SIZE})...")
            for epoch in range(1, EPOCHS + 1):
                train_acc = train(model, device, train_loader, optimizer, epoch)
                test_acc = test(model, device, test_loader, return_preds=False)
                train_acc_history.append(train_acc)
                test_acc_history.append(test_acc)
                print("-" * 60)
                
            print("\n학습 완료!")
            
            # --- 최종 결과 분석 ---
            print("\n[INFO] 최종 성능 평가 및 시각화 생성...")
            final_acc, y_pred, y_true = test(model, device, test_loader, return_preds=True)
            print(f"최종 테스트 정확도: {final_acc:.2f}%")

            plot_accuracy(train_acc_history, test_acc_history)
            
            mnist_classes = [str(i) for i in range(10)]
            plot_confusion_matrix(y_true, y_pred, classes=mnist_classes)

            plot_weight_distribution(model, fitter, title_suffix="After Training")
            
            save_accuracy_to_excel(train_acc_history, test_acc_history)
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR] 시뮬레이션 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)