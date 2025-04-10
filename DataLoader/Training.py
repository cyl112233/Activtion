import tqdm
import torch
from dataloader import get_dataloader
from model_buidiling.modle import DenseNet121, VGG16_model, MyResNet50
from Mode import VisionTransformer
from Val import compute
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from EELU_B import APT,CPAF

epochs = 100
BATCH_SIZE = 64
save_function = "CPAF"
sava_data = "CIFAR10"
moudle = "16"
save_data = f"/Y_L/fei/Activtion/Activtion/Save_Data/{sava_data}_{save_function}_{moudle}.txt"  # 保存路径
save = open(save_data, "w")
save.write(save_function)
save.write("\n")

# 数据加载
Train, Test, LenData, num_class = get_dataloader('CIFAR10', BATCH_SIZE,
                                                file="/Y_L/fei/Activtion/Activtion/DataLoader/Data1")  # 数据集下载保存地址
# model = VisionTransformer(num_classes=10, depth=8,img_size=32, patch_size=16, in_channels=3 ,
#                  embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1,activation=torch.nn.SiLU())
# print(model)
# model.patch_embed.conv[0] = torch.nn.Conv2d(1,3,1,16)
# model = MyResNet50(num_class, activation=APT())
model = VGG16_model(num_class,activation=CPAF()).VGG16()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 损失函数和优化器
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 保存最佳权重的变量
best_accuracy = 0.0
best_model_path = f"/Y_L/fei/Activtion/Activtion/Save_Data/{sava_data}_{save_function}_{moudle}_best.pt"

for i in tqdm.tqdm(range(epochs)):
    loss_sum = 0
    Precision_sum = 0
    Recall_sum = 0
    Accuracy_sum = 0
    model.train()
    model.to(device)

    for image, label in Train:
        image = image.to(device)
        label = label.to(device)
        output = model(image)

        # 计算损失
        loss = loss_func(output, label)

        # 优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        label, output = label.cpu(), output.cpu()
        Precision, Recall, Accuracy = compute(label, output.argmax(1), num_class)
        Precision_sum += Precision
        Recall_sum += Recall
        Accuracy_sum += Accuracy

    save_train = {"Train_epoch": i + 1, "Acc": float(Accuracy_sum / LenData[0]),
                  "Loss": loss_sum / LenData[0],
                  "Precision": float(Precision_sum / LenData[0]),
                  "Recall": float(Recall_sum / LenData[0])
                  }
    save.write(str(save_train))
    save.write("\n")

    # 验证阶段
    with torch.no_grad():
        loss_test_sum = 0
        Precision_test_sum = 0
        Recall_test_sum = 0
        Accuracy_test_sum = 0
        model.eval()

        for image, label in Test:
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            # 计算损失
            loss = loss_func(output, label)
            loss_test_sum += loss.item()
            label, output = label.cpu(), output.cpu()
            Precision, Recall, Accuracy = compute(label, output.argmax(1), num_class)
            Precision_test_sum += Precision
            Recall_test_sum += Recall
            Accuracy_test_sum += Accuracy

        current_accuracy = float(Accuracy_test_sum / LenData[1])
        save_test = {"Test_epoch": i + 1, "Acc": current_accuracy,
                     "Loss": loss_test_sum / LenData[1],
                     "Precision": float(Precision_test_sum / LenData[1]),
                     "Recall": float(Recall_test_sum / LenData[1])
                     }
        save.write(str(save_test))
        save.write("\n")

        # 检查是否是最佳模型
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model, best_model_path)
            print(f"Epoch {i + 1}: New best model saved with accuracy {best_accuracy:.4f}")

save.close()
print(f"Best model saved at: {best_model_path}")


