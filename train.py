import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from preprocessing.data_loader import DEAPDataLoader
from preprocessing.dataset import create_subject_dependent_loaders, create_trial_based_loaders
from models.eeg_emotion_model import EEGEmotionRecognitionModel
from models.mobilenetv2 import MobileNetV2_Feature_Extractor
from models.capsule_network import EmotionCapsuleNet
from models.loss import MarginLoss

RESULTS_DIR = 'results'
DATA_DIR = 'results/data'
CHECKPOINT_FILE = 'results/data/checkpoint.json'


class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.002):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def should_stop(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)

    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        total_loss = 0
        valid_batches = 0
        all_preds = []
        all_labels = []

        for data, labels in train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            output = model(data)

            if isinstance(output, tuple):
                _, probs = output
            else:
                probs = output

            loss = criterion(probs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(valid_batches, 1)
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        return avg_loss, accuracy

    def evaluate(self, model, test_loader, criterion):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    _, probs = output
                else:
                    probs = output

                loss = criterion(probs, labels)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(test_loader), 1)
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0.0
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

        return avg_loss, accuracy, f1, cm

    def save_json(self, data, filename):
        filepath = f'{DATA_DIR}/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def train_single_fold(self, model, train_loader, test_loader,
                          criterion, num_epochs, lr, weight_decay):
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

        early_stopping = EarlyStopping(patience=15, min_delta=0.001)

        history = {
            'epochs': [], 'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [], 'test_f1': []
        }

        best_acc = 0
        best_cm = None

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer
            )
            scheduler.step()
            test_loss, test_acc, test_f1, cm = self.evaluate(
                model, test_loader, criterion
            )

            history['epochs'].append(epoch + 1)
            history['train_loss'].append(round(train_loss, 6))
            history['train_acc'].append(round(train_acc, 6))
            history['test_loss'].append(round(test_loss, 6))
            history['test_acc'].append(round(test_acc, 6))
            history['test_f1'].append(round(test_f1, 6))

            if test_acc > best_acc:
                best_acc = test_acc
                best_cm = cm

            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
                print(f"      Epoch {epoch + 1}/{num_epochs}: "
                      f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                      f"Test Acc={test_acc:.4f}, F1={test_f1:.4f}")

            if early_stopping.should_stop(test_acc):
                print(f"      ⏹ Early stopping at epoch {epoch + 1} "
                      f"(best test acc: {best_acc:.4f})")
                break

        return history, best_acc, best_cm

    def train_subject(self, subject_id, segments, labels, emotion_dim,
                      model_name='full_model', create_model_fn=None):
        print(f"    Subject: s{subject_id:02d}, Dim: {emotion_dim}, Model: {model_name}")

        unique, counts = np.unique(labels, return_counts=True)
        print(f"    标签分布: {dict(zip(unique.tolist(), counts.tolist()))}")

        fold_accuracies = []
        total_cm = np.zeros((2, 2), dtype=int)
        n_splits = self.config.get('n_splits', 10)

        for fold_idx in range(n_splits):
            print(f"\n    Fold {fold_idx + 1}/{n_splits}")

            train_loader, test_loader = create_trial_based_loaders(
                segments, labels,
                batch_size=self.config.get('batch_size', 64),
                n_splits=n_splits,
                fold_idx=fold_idx
            )

            model = create_model_fn().to(self.device)

            if model_name in ['full_model', 'no_se']:
                criterion = MarginLoss(m_plus=0.9, m_minus=0.1, lambda_val=0.5)
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

            history, best_acc, best_cm = self.train_single_fold(
                model, train_loader, test_loader, criterion,
                num_epochs=self.config.get('num_epochs', 100),
                lr=self.config.get('lr', 1e-3),
                weight_decay=self.config.get('weight_decay', 1e-4),
            )

            self.save_json(history,
                           f'{model_name}_{emotion_dim}_s{subject_id:02d}_fold{fold_idx + 1}.json')

            fold_accuracies.append(round(best_acc, 6))
            if best_cm is not None:
                total_cm += best_cm

            print(f"      Fold {fold_idx + 1} Best Acc: {best_acc:.4f}")

        self.save_json({
            'subject_id': subject_id,
            'emotion_dim': emotion_dim,
            'model_name': model_name,
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': round(float(np.mean(fold_accuracies)), 6),
            'std_accuracy': round(float(np.std(fold_accuracies)), 6),
            'confusion_matrix': total_cm.tolist()
        }, f'{model_name}_{emotion_dim}_s{subject_id:02d}_results.json')

        return fold_accuracies, total_cm


# ============================================================
# 断点续训
# ============================================================

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_checkpoint(model_key, dim_idx, subject_id, completed_results):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'model_key': model_key, 'dim_idx': dim_idx,
            'subject_id': subject_id, 'completed_results': completed_results
        }, f, indent=2, ensure_ascii=False)


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


def load_json_file(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 模型定义
# ============================================================

def create_full_model():
    model = EEGEmotionRecognitionModel(
        num_classes=2,
        primary_caps=16, primary_dim=8,
        emotion_dim=16, routing_iterations=3,
        dropout_rate=0.3
    )
    stats = model.get_param_stats()
    print(f"    参数: 总{stats['total']:,}, "
          f"可训练{stats['trainable']:,}")
    return model


class NoSEModel(nn.Module):
    """消融：去掉SE模块"""
    def __init__(self):
        super().__init__()
        from models.eeg_emotion_model import EEGMobileNetV2
        self.feature_extractor = EEGMobileNetV2(dropout_rate=0.3)
        # 替换SE为Identity
        self.feature_extractor.se = nn.Identity()
        self.classifier = EmotionCapsuleNet(
            in_features=128, num_classes=2,
            primary_caps=16, primary_dim=8,
            emotion_dim=16, routing_iterations=3
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


class NoCapsModel(nn.Module):
    """消融：去掉胶囊网络"""
    def __init__(self):
        super().__init__()
        from models.eeg_emotion_model import EEGMobileNetV2
        self.feature_extractor = EEGMobileNetV2(dropout_rate=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class BaselineLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=64,
                            num_layers=2, batch_first=True,
                            dropout=0.2, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.squeeze(1)
        _, (h_n, _) = self.lstm(x)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(hidden)


# ============================================================
# 实验运行
# ============================================================

def run_experiment(config, data_loader, trainer, model_name, create_model_fn,
                   emotion_dims, start_dim_idx=0, start_subject=1,
                   existing_results=None):
    results = existing_results if existing_results else {}

    for dim_idx in range(start_dim_idx, len(emotion_dims)):
        emotion_dim = emotion_dims[dim_idx]

        print(f"\n{'='*60}")
        print(f"[{model_name}] Dimension: {emotion_dim.upper()}")
        print(f"{'='*60}")

        if emotion_dim in results and dim_idx == start_dim_idx:
            subject_accs = results[emotion_dim]['subject_accuracies']
            total_cm = np.array(results[emotion_dim].get(
                'confusion_matrix', [[0, 0], [0, 0]]))
            s_start = start_subject
        else:
            subject_accs = []
            total_cm = np.zeros((2, 2), dtype=int)
            s_start = 1

        for subject_id in range(s_start, 33):
            print(f"\n  Subject s{subject_id:02d} - {emotion_dim}")

            segments, binary_labels = data_loader.preprocess_subject(subject_id)
            labels = binary_labels[emotion_dim]

            fold_accs, cm = trainer.train_subject(
                subject_id, segments, labels, emotion_dim,
                model_name=model_name,
                create_model_fn=create_model_fn
            )
            mean_acc = np.mean(fold_accs)
            subject_accs.append(round(mean_acc, 6))
            total_cm += cm
            print(f"    Mean Acc: {mean_acc:.4f}")

            results[emotion_dim] = {
                'subject_accuracies': subject_accs,
                'mean_accuracy': round(float(np.mean(subject_accs)), 6),
                'std_accuracy': round(float(np.std(subject_accs)), 6),
                'confusion_matrix': total_cm.tolist()
            }
            trainer.save_json(results, f'{model_name}_all_results.json')

            next_subject = subject_id + 1
            next_dim_idx = dim_idx
            if next_subject > 32:
                next_subject = 1
                next_dim_idx = dim_idx + 1
            save_checkpoint(model_name, next_dim_idx, next_subject, results)

        print(f"\n  [{model_name}] {emotion_dim.upper()}: "
              f"{np.mean(subject_accs):.4f} ± {np.std(subject_accs):.4f}")

    return results


def main():
    config = {
        'data_dir': './data/DEAP/data_preprocessed_python/',
        'batch_size': 64,
        'num_epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'n_splits': 10,
        'emotion_dimensions': ['valence', 'arousal', 'dominance'],
    }

    experiments = {
        'full_model': {'name': '完整模型 (SE+MobileNetV2+CapsNet)',
                       'create_fn': create_full_model},
    }

    experiment_order = list(experiments.keys())
    emotion_dims = config['emotion_dimensions']

    data_loader = DEAPDataLoader(config['data_dir'])
    trainer = Trainer(config)

    checkpoint = load_checkpoint()
    if checkpoint:
        resume_model = checkpoint['model_key']
        resume_dim_idx = checkpoint['dim_idx']
        resume_subject = checkpoint['subject_id']
        resume_results = checkpoint['completed_results']

        print("=" * 60)
        print("⚡ 从断点继续")
        print(f"   模型: {resume_model}")
        dim_name = emotion_dims[resume_dim_idx] if resume_dim_idx < len(emotion_dims) else "已完成"
        print(f"   维度: {dim_name}")
        print(f"   受试者: s{resume_subject:02d}")
        print("=" * 60)
        start_exp_idx = experiment_order.index(resume_model)
    else:
        print("=" * 60)
        print("🚀 开始全新训练")
        print("=" * 60)
        start_exp_idx = 0
        resume_dim_idx = 0
        resume_subject = 1
        resume_results = None

    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    all_experiment_results = {}

    for exp_idx in range(start_exp_idx, len(experiment_order)):
        model_key = experiment_order[exp_idx]
        exp_info = experiments[model_key]

        print(f"\n{'#'*60}")
        print(f"# Experiment: {exp_info['name']}")
        print(f"{'#'*60}")

        if exp_idx == start_exp_idx and checkpoint:
            s_dim_idx = resume_dim_idx
            s_subject = resume_subject
            s_results = resume_results
        else:
            s_dim_idx = 0
            s_subject = 1
            s_results = None

        if s_dim_idx >= len(emotion_dims):
            loaded = load_json_file(f'{DATA_DIR}/{model_key}_all_results.json')
            if loaded:
                all_experiment_results[model_key] = loaded
            continue

        results = run_experiment(
            config, data_loader, trainer,
            model_name=model_key,
            create_model_fn=exp_info['create_fn'],
            emotion_dims=emotion_dims,
            start_dim_idx=s_dim_idx,
            start_subject=s_subject,
            existing_results=s_results
        )
        all_experiment_results[model_key] = results

    summary = {}
    for model_key, results in all_experiment_results.items():
        summary[model_key] = {
            'display_name': experiments[model_key]['name'],
            'dimensions': {}
        }
        for dim, data in results.items():
            summary[model_key]['dimensions'][dim] = {
                'mean_accuracy': data['mean_accuracy'],
                'std_accuracy': data['std_accuracy'],
                'confusion_matrix': data['confusion_matrix']
            }

    trainer.save_json(summary, 'experiment_summary.json')
    if 'full_model' in all_experiment_results:
        trainer.save_json(all_experiment_results['full_model'], 'all_results.json')

    clear_checkpoint()

    print(f"\n{'='*60}")
    print("✅ ALL EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print(f"{'模型':<35} {'维度':<12} {'准确率':<20}")
    print(f"{'-'*60}")
    for model_key, results in all_experiment_results.items():
        name = experiments[model_key]['name']
        for dim, data in results.items():
            print(f"  {name:<33} {dim.upper():<10} "
                  f"{data['mean_accuracy']:.4f} ± {data['std_accuracy']:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()