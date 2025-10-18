import torch
import time
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime


class TrainingMonitor:
    """训练监控器"""

    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.losses = []
        self.learning_rates = []
        self.timestamps = []
        self.start_time = time.time()

    def log_step(self, loss, lr, step):
        """记录训练步"""
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.timestamps.append(time.time() - self.start_time)

        # 每100步保存一次
        if step % 100 == 0:
            self.save_progress(step)

    def save_progress(self, step):
        """保存训练进度"""
        progress = {
            'step': step,
            'losses': self.losses,
            'learning_rates': self.learning_rates,
            'timestamps': self.timestamps,
            'average_loss': sum(self.losses[-100:]) / min(100, len(self.losses))
        }

        # 保存JSON
        with open(f"{self.log_dir}/progress_step_{step}.json", 'w') as f:
            json.dump(progress, f, indent=2)

        # 绘制损失曲线
        if len(self.losses) > 10:
            self.plot_progress(step)

    def plot_progress(self, step):
        """绘制训练进度图"""
        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True)

        # 学习率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('LR')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/training_progress_step_{step}.png")
        plt.close()


def create_mini_batch_trainer():
    """创建小批量训练器"""

    class MiniBatchTrainer:
        def __init__(self, model, optimizer, max_batches=1000):
            self.model = model
            self.optimizer = optimizer
            self.max_batches = max_batches
            self.monitor = TrainingMonitor()

        def train(self, dataloader, epochs=3):
            """限制批次数量的训练"""
            self.model.train()
            global_step = 0

            for epoch in range(epochs):
                total_loss = 0
                batch_count = 0

                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= self.max_batches:
                        break

                    loss = self.model(
                        batch['input_ids'],
                        batch['attention_mask'],
                        batch['labels'],
                        batch['mask']
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1
                    global_step += 1

                    # 监控
                    self.monitor.log_step(loss.item(), 2e-5, global_step)

                    if batch_idx % 20 == 0:
                        avg_loss = total_loss / (batch_idx + 1)
                        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}')

                avg_epoch_loss = total_loss / batch_count
                print(f'Epoch {epoch} 完成, 平均Loss: {avg_epoch_loss:.4f}')

    return MiniBatchTrainer


# 使用示例
if __name__ == '__main__':
    # 测试监控器
    monitor = TrainingMonitor()
    for i in range(500):
        monitor.log_step(1.0 / (i + 1), 2e-5, i)