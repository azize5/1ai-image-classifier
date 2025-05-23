
# Yapay Zeka Destekli Görüntü Sınıflandırıcı

Bu proje, 10 farklı hayvan sınıfını tanıyabilen bir görüntü sınıflandırma modelini içerir. PyTorch ile geliştirilmiş olan bu model, kullanıcıların yüklediği görselleri analiz ederek, doğru sınıfa ait olup olmadığını tahmin eder.

## İçerik

- **`train.py`**: PyTorch ile veri ön işleme, augmentation, model eğitimi, early stopping ve metrik takibi işlevlerini yerine getirir.
- **`app.py`**: Gradio ile geliştirilmiş kullanıcı dostu arayüz. Görsel yükleme, tahmin yapma ve sınıf çıktısı sunma işlemleri için kullanılır.
- **`models/best_model.pth`**: Eğitilmiş modelin kaydedildiği dosya.
- **`raw-img/`**: 10 farklı sınıftan oluşan görsel verilerin bulunduğu klasör 5 sınıf kullanıldı.

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

Modelin başarımını değerlendirirken kullanılan başlıca metrikler ve sonuçları:

- **Accuracy**: 0.9489
- **Precision**: 0.9489
- **Recall**: 0.9489
- **F1 socore: 0.9487

  eklediğim görsellerde modelin zaten hangi sınıf olduğu görünmektedir.

Bu metrikler, modelin genel başarısını ve her bir sınıf için ne kadar başarılı olduğunu gösterir.

-----------------------------------------
           ÖNEMLİ!!!!!!!!! 
--------------------------------------------
Hocam ben githup'a projeyi yüklemekte cok zorluk çektim ve yükleyemedim arkadaşlarımla da bu konu hakkında baya çabaladık ama bir şekilde olmadı yani yükleyebildiğim kadarını yüklemeye çalıştım zaten demo videosu ve görsellerden de anlaşılacağı üzere projem eksiksiz çalışmaktadır. best_model.pth dosyamı yükleyemedim boyutu çok büyük olduğu için(128 mb) yani hatanın bu olduğunu gösteriyordu. 
