import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime


def main():
    # Başlangıç zamanını kaydet
    start_time = time.time()

    # Yapılandırma ayarları
    config = {
        "data_dir": "./raw-img",
        "batch_size": 32,
        "num_epochs": 20,
        "learning_rate": 0.001,
        "patience": 5,
        "img_size": 224,
        "model_name": "resnet18",
        "val_split": 0.2,
        "use_pretrained": True
    }

    # Cihaz seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    # Klasörleri oluştur
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # Veri dönüşümleri tanımlama
    print("Veri dönüşümleri ayarlanıyor...")
    train_transforms = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        RandomResizedCrop(config['img_size'], scale=(0.8, 1.0)),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Veri yükleme
    print(f"Veri yükleniyor: {config['data_dir']}")
    try:
        dataset = ImageFolder(root=config['data_dir'], transform=train_transforms)
        class_names = dataset.classes
        print(f"Sınıflar: {class_names}")
        print(f"Toplam görüntü sayısı: {len(dataset)}")

        # Kaç görüntü var sınıf başına kontrol et
        class_counts = {class_name: 0 for class_name in class_names}
        for _, label in dataset:
            class_counts[class_names[label]] += 1

        print("Sınıf başına görüntü sayısı:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")

        # Veri setlerini böl
        val_size = int(config['val_split'] * len(dataset))
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Dönüşümleri uygula
        train_dataset.dataset.transform = train_transforms
        val_dataset.dataset.transform = val_transforms

        # DataLoader'ları oluştur
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

        print(f"Eğitim seti: {len(train_dataset)} görüntü")
        print(f"Doğrulama seti: {len(val_dataset)} görüntü")

    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        return

    # Model oluşturma
    print(f"Model oluşturuluyor: {config['model_name']}")
    try:
        if config['model_name'] == 'resnet18':
            model = models.resnet18(pretrained=config['use_pretrained'])
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        elif config['model_name'] == 'resnet50':
            model = models.resnet50(pretrained=config['use_pretrained'])
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        elif config['model_name'] == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=config['use_pretrained'])
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
        else:
            print("Desteklenmeyen model türü, varsayılan olarak ResNet18 kullanılıyor.")
            model = models.resnet18(pretrained=config['use_pretrained'])
            model.fc = nn.Linear(model.fc.in_features, len(class_names))

        model = model.to(device)
    except Exception as e:
        print(f"Model oluşturulurken hata oluştu: {e}")
        return

    # Kayıp fonksiyonu ve optimizer tanımlama
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    try:
        # Yeni PyTorch sürümleri için
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    except TypeError:
        # Eski PyTorch sürümleri için verbose parametresi olmadan
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        print("Eski PyTorch sürümü tespit edildi, verbose parametresi atlandı.")

    # TensorBoard yazıcısı
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"./logs/run_{timestamp}")

    # Early stopping için değişkenler
    best_val_loss = float('inf')
    best_val_acc = 0.0
    counter = 0
    best_epoch = 0
    best_model_path = f"./models/best_model_{timestamp}.pth"
    final_model_path = f"./models/final_model_{timestamp}.pth"

    # Metrik kayıtları
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Eğitim döngüsü
    print(f"\nEğitim başlıyor: {config['num_epochs']} epoch")
    try:
        for epoch in range(config['num_epochs']):
            # Eğitim aşaması
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Gradyanları sıfırla
                optimizer.zero_grad()

                # İleri geçiş
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Geri yayılım
                loss.backward()
                optimizer.step()

                # İstatistikleri güncelle
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # Her 10 batch'te ilerleme durumunu göster
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Batch [{i + 1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")

            # Epoch istatistiklerini hesapla
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100 * train_correct / train_total
            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc)

            # Doğrulama aşaması
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Doğrulama istatistiklerini hesapla
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100 * val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)

            # Öğrenme oranını güncelle
            scheduler.step(epoch_val_loss)

            # TensorBoard'a yaz
            writer.add_scalars('Loss', {'train': epoch_train_loss, 'val': epoch_val_loss}, epoch)
            writer.add_scalars('Accuracy', {'train': epoch_train_acc, 'val': epoch_val_acc}, epoch)

            # Çıktıları yazdır
            print(f"Epoch [{epoch + 1}/{config['num_epochs']}] - "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

            # Early stopping kontrol et
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_val_loss = epoch_val_loss
                counter = 0
                best_epoch = epoch

                # En iyi modeli kaydet
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_acc': epoch_val_acc,
                    'classes': class_names
                }, best_model_path)
                print(f"En iyi model kaydedildi (Doğrulama Doğruluğu: {best_val_acc:.2f}%)")
            else:
                counter += 1
                if counter >= config['patience']:
                    print(f"Early stopping uygulandı (En iyi epoch: {best_epoch + 1})")
                    break

        # Son modeli kaydet
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc,
            'classes': class_names
        }, final_model_path)
        print(f"Son model kaydedildi: {final_model_path}")

        # En iyi modeli yükle ve değerlendir
        print("\nEn iyi model değerlendiriliyor...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Doğrulama seti üzerinde değerlendirme
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Performans metrikleri hesaplama
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Sınıflandırma raporu
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True,
                                       zero_division=0)

        # Sonuçları kaydet
        results = {
            "config": config,
            "best_epoch": best_epoch + 1,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
            "final_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "class_metrics": report,
            "class_names": class_names,
            "training_time": time.time() - start_time
        }

        with open(f"./results/training_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=4)

        # Eğitim sürecini görselleştir
        plt.figure(figsize=(12, 5))

        # Kayıp grafiği
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs. Epoch')

        # Doğruluk grafiği
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy vs. Epoch')

        plt.tight_layout()
        plt.savefig(f"./results/training_plot_{timestamp}.png")

        # Uygulama için sınıf isimlerini ve model yolunu kaydet
        model_info = {
            "model_path": best_model_path,
            "class_names": class_names,
            "img_size": config['img_size']
        }

        with open("model_info.json", "w") as f:
            json.dump(model_info, f, indent=4)

        # Uygulama için kullanım kolaylığı açısından best_model.pth olarak da kopyala
        import shutil
        shutil.copy(best_model_path, "best_model.pth")

        print("\nEğitim tamamlandı.")
        print(f"Toplam süre: {(time.time() - start_time) / 60:.2f} dakika")
        print(f"En iyi doğrulama doğruluğu: {best_val_acc:.2f}% (Epoch {best_epoch + 1})")
        print(f"Son metrikler:")
        print(f"  - Doğruluk: {accuracy:.4f}")
        print(f"  - Hassasiyet: {precision:.4f}")
        print(f"  - Duyarlılık: {recall:.4f}")
        print(f"  - F1 Skoru: {f1:.4f}")
        print(f"\nDetaylı sonuçlar './results/training_results_{timestamp}.json' dosyasına kaydedildi.")
        print(f"Eğitim grafikleri './results/training_plot_{timestamp}.png' dosyasına kaydedildi.")
        print(f"Model 'best_model.pth' olarak kaydedildi.")

    except Exception as e:
        print(f"Eğitim sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()