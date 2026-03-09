import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import os

from preprocessing.data_loader import DEAPDataLoader
from preprocessing.dataset import create_subject_dependent_loaders
from models.eeg_emotion_model import EEGEmotionRecognitionModel
from models.loss import MarginLoss

from train import Trainer, EarlyStopping

SINGLE_SUBJECT_DIR = 'results/data/single_subject'


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


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_single_subject(subject_id, emotion_dim, config):
    os.makedirs(SINGLE_SUBJECT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  单被试实验 - Subject s{subject_id:02d} | Dim: {emotion_dim.upper()}")
    print(f"{'='*60}")

    data_loader = DEAPDataLoader(config['data_dir'])
    trainer = Trainer(config)

    segments, binary_labels = data_loader.preprocess_subject(subject_id)
    labels = binary_labels[emotion_dim]

    print(f"  数据形状: {segments.shape}, 标签形状: {labels.shape}")

    fold_accuracies = []
    fold_f1_scores = []
    total_cm = np.zeros((2, 2), dtype=int)
    n_splits = config.get('n_splits', 10)

    for fold_idx in range(n_splits):
        print(f"\n  Fold {fold_idx + 1}/{n_splits}")

        train_loader, test_loader = create_subject_dependent_loaders(
            segments, labels,
            batch_size=config.get('batch_size', 64),
            n_splits=n_splits,
            fold_idx=fold_idx
        )

        model = create_full_model().to(trainer.device)
        criterion = MarginLoss(
            m_plus=config.get('m_plus', 0.9),
            m_minus=config.get('m_minus', 0.1),
            lambda_val=config.get('lambda_val', 0.5)
        )

        history, best_acc, best_cm = trainer.train_single_fold(
            model, train_loader, test_loader, criterion,
            num_epochs=config.get('num_epochs', 40),
            lr=config.get('lr', 5e-4),
            weight_decay=config.get('weight_decay', 5e-4),
        )

        history_path = os.path.join(
            SINGLE_SUBJECT_DIR,
            f'single_subject_{emotion_dim}_s{subject_id:02d}_fold{fold_idx + 1}.json'
        )
        save_json(history, history_path)

        # Retrieve the F1 score from the same epoch that produced best_acc
        if history['test_acc']:
            best_epoch_idx = int(np.argmax(history['test_acc']))
            best_f1 = history['test_f1'][best_epoch_idx] if history['test_f1'] else 0.0
        else:
            best_f1 = 0.0
        fold_accuracies.append(round(best_acc, 6))
        fold_f1_scores.append(round(best_f1, 6))
        if best_cm is not None:
            total_cm += best_cm

        print(f"  Fold {fold_idx + 1} Best Acc: {best_acc:.4f}, Best F1: {best_f1:.4f}")

    mean_acc = round(float(np.mean(fold_accuracies)), 6)
    std_acc = round(float(np.std(fold_accuracies)), 6)
    mean_f1 = round(float(np.mean(fold_f1_scores)), 6)
    std_f1 = round(float(np.std(fold_f1_scores)), 6)

    subject_result = {
        'subject_id': subject_id,
        'emotion_dim': emotion_dim,
        'fold_accuracies': fold_accuracies,
        'fold_f1_scores': fold_f1_scores,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'confusion_matrix': total_cm.tolist()
    }

    result_path = os.path.join(
        SINGLE_SUBJECT_DIR,
        f'single_subject_{emotion_dim}_s{subject_id:02d}_results.json'
    )
    save_json(subject_result, result_path)

    print(f"\n  ✅ Subject s{subject_id:02d} [{emotion_dim.upper()}]: "
          f"Acc={mean_acc:.4f} ± {std_acc:.4f}, "
          f"F1={mean_f1:.4f} ± {std_f1:.4f}")

    return subject_result


def main():
    parser = argparse.ArgumentParser(
        description='单被试十折交叉验证实验 (Subject-Dependent 10-Fold CV)'
    )
    parser.add_argument('--subject', '-s', type=int, default=None,
                        help='受试者编号 (1-32)，不指定则运行所有32个受试者'
                             '（注意：运行所有受试者需要大量时间，共320折训练）')
    parser.add_argument('--dim', '-d', type=str, default='valence',
                        choices=['valence', 'arousal', 'dominance'],
                        help='情绪维度 (默认: valence)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='训练轮数 (默认: 40)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小 (默认: 64)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率 (默认: 5e-4)')
    args = parser.parse_args()

    config = {
        'data_dir': './data/DEAP/data_preprocessed_python/',
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 5e-4,
        'n_splits': 10,
    }

    emotion_dim = args.dim

    if args.subject is not None:
        if not (1 <= args.subject <= 32):
            parser.error('受试者编号必须在 1-32 之间')
        subjects = [args.subject]
    else:
        subjects = list(range(1, 33))

    print('=' * 60)
    print('🚀 单被试十折交叉验证实验')
    print('=' * 60)
    print(f'  情绪维度  : {emotion_dim.upper()}')
    print(f'  受试者    : {subjects}')
    print(f'  训练轮数  : {config["num_epochs"]}')
    print(f'  批大小    : {config["batch_size"]}')
    print(f'  学习率    : {config["lr"]}')
    print(f'  结果目录  : {SINGLE_SUBJECT_DIR}')
    print('=' * 60)

    all_results = []
    for subject_id in subjects:
        result = run_single_subject(subject_id, emotion_dim, config)
        all_results.append(result)

    all_accs = [r['mean_accuracy'] for r in all_results]
    all_f1s = [r['mean_f1'] for r in all_results]

    print(f"\n{'='*60}")
    print(f"✅ 实验完成 | 维度: {emotion_dim.upper()}")
    print(f"   受试者数: {len(all_results)}")
    print(f"   平均准确率: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")
    print(f"   平均 F1  : {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
    print(f"{'='*60}")

    summary_path = os.path.join(
        SINGLE_SUBJECT_DIR,
        f'single_subject_{emotion_dim}_summary.json'
    )
    save_json({
        'emotion_dim': emotion_dim,
        'subjects': subjects,
        'subject_results': all_results,
        'overall_mean_accuracy': round(float(np.mean(all_accs)), 6),
        'overall_std_accuracy': round(float(np.std(all_accs)), 6),
        'overall_mean_f1': round(float(np.mean(all_f1s)), 6),
        'overall_std_f1': round(float(np.std(all_f1s)), 6),
    }, summary_path)
    print(f"  汇总结果已保存: {summary_path}")


if __name__ == '__main__':
    main()
