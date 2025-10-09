import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

class LetterPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LetterPerceptron, self).__init__()
        # 定义一个简单的多层感知机
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

def prepare_data():
    """
    准备示例数据
    返回:
        features: 21x63的输入特征 (7个字母的三种编码)
        labels: 21x7的标签 (每个位置对应哪个字母)
    """
    # 生成示例数据 - 在实际应用中，这里应该替换为你的真实数据
    # 假设有7个不同的字母，每个字母有3种编码方式，每种编码长度为3
    num_samples = 21
    num_letters = 7
    encoding_length = 3
    
    # 生成21x63的特征矩阵
    features = np.random.randn(num_samples, num_letters * encoding_length * 3)
    
    # 生成21x7的标签矩阵，每行是一个one-hot编码，表示对应的字母
    labels = np.zeros((num_samples, num_letters))
    for i in range(num_samples):
        letter_idx = i % num_letters  # 循环使用7个字母
        labels[i, letter_idx] = 1
    
    return torch.FloatTensor(features), torch.FloatTensor(labels)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    训练模型
    """
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, actual = torch.max(labels.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == actual).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                _, actual = torch.max(labels.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == actual).sum().item()
        
        # 计算平均损失和准确率
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.2f}%')
            print('-' * 50)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    绘制训练历史
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # 参数设置
    input_size = 63  # 21x63输入，但我们将展平为63维向量
    hidden_size = 128
    output_size = 7  # 7个字母
    learning_rate = 0.001
    batch_size = 8
    num_epochs = 100
    
    # 准备数据
    print("准备数据...")
    features, labels = prepare_data()
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = LetterPerceptron(input_size, hidden_size, output_size)
    print(f"模型结构:\n{model}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs
    )
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 测试模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_val)
        _, predicted = torch.max(test_outputs, 1)
        _, actual = torch.max(y_val, 1)
        accuracy = (predicted == actual).float().mean()
        
        print(f"\n最终验证集准确率: {accuracy.item()*100:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), 'letter_perceptron.pth')
    print("模型已保存为 'letter_perceptron.pth'")

# 预测函数
def predict_letter(model, input_data):
    """
    使用训练好的模型预测字母
    """
    model.eval()
    with torch.no_grad():
        if isinstance(input_data, list):
            input_data = torch.FloatTensor(input_data)
        
        # 确保输入数据形状正确
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)
        
        output = model(input_data)
        _, predicted = torch.max(output, 1)
        
        return predicted.item()

if __name__ == "__main__":
    main()
    
    # 示例：如何使用预测函数
    print("\n示例预测:")
    # 加载训练好的模型
    model = LetterPerceptron(63, 128, 7)
    model.load_state_dict(torch.load('letter_perceptron.pth'))
    
    # 创建一个随机测试样本
    test_sample = np.random.randn(63)
    predicted_letter = predict_letter(model, test_sample)
    print(f"预测的字母索引: {predicted_letter}")