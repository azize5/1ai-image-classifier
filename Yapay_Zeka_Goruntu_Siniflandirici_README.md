
# Yapay Zeka Destekli Görüntü Sınıflandırıcı

Bu proje, 10 farklı hayvan sınıfını tanıyabilen bir görüntü sınıflandırma modelini içerir. PyTorch ile geliştirilmiş olan bu model, kullanıcıların yüklediği görselleri analiz ederek, doğru sınıfa ait olup olmadığını tahmin eder.

## İçerik

- **`train.py`**: PyTorch ile veri ön işleme, augmentation, model eğitimi, early stopping ve metrik takibi işlevlerini yerine getirir.
- **`app.py`**: Gradio ile geliştirilmiş kullanıcı dostu arayüz. Görsel yükleme, tahmin yapma ve sınıf çıktısı sunma işlemleri için kullanılır.
- **`models/best_model.pth`**: Eğitilmiş modelin kaydedildiği dosya.
- **`raw-img/`**: 10 farklı sınıftan oluşan görsel verilerin bulunduğu klasör (kullanıcı tarafından sağlanan veriler burada yer alır).

## Kullanılan Teknikler

- **Veri Ön İşleme**: Görsellerin normalize edilmesi ve uygun boyutlara yeniden dönüştürülmesi.
- **Augmentation**: Görseller üzerinde yatay çevirme (horizontal flip), döndürme (rotation) ve kırpma (crop) gibi dönüşümler.
- **Model**: ResNet-18 mimarisi + Dropout katmanı.
- **Overfitting Önleme**: Dropout, ağırlık çürümesi (weight decay) ve EarlyStopping (patience=5) kullanımı ile aşırı öğrenme (overfitting) engellenir.
- **Metrikler**: Accuracy (doğruluk), Precision (kesinlik), Recall (duyarlılık) gibi metrikler ile modelin başarısı ölçülür.

## Gereksinimler

Projeyi çalıştırmadan önce aşağıdaki bağımlılıkları yüklemeniz gerekir:

```bash
pip install -r requirements.txt
```

### `requirements.txt` dosyası

- torch
- torchvision
- gradio
- numpy
- matplotlib
- scikit-learn

## Eğitim

Modeli eğitmek için aşağıdaki komutu kullanabilirsiniz:

```bash
python train.py
```

Bu komut, modelin eğitim sürecini başlatacak ve eğitim tamamlandıktan sonra en iyi model `models/best_model.pth` dosyasına kaydedilecektir.

## Arayüz Kullanımı

Modeli çalıştıran ve görselleri sınıflandıran arayüzü başlatmak için aşağıdaki komutu kullanabilirsiniz:

```bash
python app.py
```

Bu komut, Gradio arayüzünü açarak kullanıcıların görsel yükleyip, modelin tahminini görmelerine olanak sağlar.

Arayüzde aşağıdaki adımlar takip edilebilir:

1. Görsel yükleyin.
2. Model, görseli analiz ederek hangi sınıfa ait olduğunu tahmin eder.
3. Tahmin edilen sınıf ve güven skoru kullanıcıya gösterilir.

## Eğitim Süreci

### Veri Ön İşleme
Görseller, model için uygun hale gelmesi amacıyla normalize edilir ve yeniden boyutlandırılır. Bu adımda kullanılan parametreler:
- **Resize**: Görseller 224x224 boyutlarına dönüştürülür.
- **Normalization**: Görsellerin piksel değerleri [0, 1] aralığına normalize edilir.

### Augmentation
Veri setinin çeşitliliğini artırmak amacıyla görsel augmentation teknikleri kullanılır:
- **Horizontal Flip**: Görsellerin yatayda ters çevrilmesi.
- **Rotation**: Görsellerin rastgele döndürülmesi.
- **Random Crop**: Görsellerin rastgele kırpılması.

### Model
Model olarak ResNet-18 kullanılmaktadır. Bu mimari, derin öğrenme için güçlü ve hızlı bir modeldir. Ayrıca, aşırı öğrenmeyi (overfitting) engellemek için Dropout katmanı ve ağırlık çürümesi (weight decay) uygulanır.

### Eğitim ve Erken Durdurma (Early Stopping)
Modelin eğitiminde erken durdurma (early stopping) tekniği kullanılarak, modelin erken aşamalarda aşırı öğrenmeye başlaması engellenir. Bu, eğitim süresinin optimize edilmesine yardımcı olur.

## Sonuçlar

Modelin başarımını değerlendirirken kullanılan başlıca metrikler:

- **Accuracy**: Modelin doğru tahmin ettiği örneklerin toplam örneklere oranı.
- **Precision**: Modelin doğru pozitif tahminlerinin, toplam pozitif tahminlere oranı.
- **Recall**: Modelin doğru pozitif tahminlerinin, toplam gerçek pozitif örneklere oranı.

Bu metrikler, modelin genel başarısını ve her bir sınıf için ne kadar başarılı olduğunu gösterir.
