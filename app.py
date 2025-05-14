
import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # GUI olmadan backend kullan

# Cihaz seçimi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

# Model bilgilerini yükleme
try:
    if os.path.exists("model_info.json"):
        with open("model_info.json", "r") as f:
            model_info = json.load(f)
        model_path = model_info["model_path"]
        classes = model_info["class_names"]
        img_size = model_info["img_size"]
        print(f"Model bilgileri yüklendi: {model_path}")
        print(f"Sınıflar: {classes}")
    else:
        # Model bilgileri yoksa varsayılan değerleri kullan
        model_path = "best_model.pth"
        classes = sorted(os.listdir("./raw-img"))
        img_size = 224
        print(f"Model bilgi dosyası bulunamadı, varsayılan değerler kullanılıyor.")
        print(f"Varsayılan model yolu: {model_path}")
        print(f"Tespit edilen sınıflar: {classes}")
except Exception as e:
    print(f"Model bilgileri yüklenirken hata oluştu: {e}")
    model_path = "best_model.pth"
    classes = ["Sınıf bilgisi yüklenemedi"]
    img_size = 224


# Model yükleme
def load_model():
    try:
        print(f"Model yükleniyor: {model_path}")
        if os.path.exists(model_path):
            # Checkpoint formatında kaydedilmiş modeli yükleme
            checkpoint = torch.load(model_path, map_location=device)

            # Eğer checkpoint bir sözlük ise
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
                if "classes" in checkpoint:
                    global classes
                    classes = checkpoint["classes"]
                    print(f"Checkpoint'ten sınıf isimleri güncellendi: {classes}")
            else:
                model_state = checkpoint

            # Model mimarisini belirle (ResNet18 varsayılan)
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

            # Model durumunu yükle
            model.load_state_dict(model_state)
            model = model.to(device)
            model.eval()
            print("Model başarıyla yüklendi.")
            return model
        else:
            print(f"Model dosyası bulunamadı: {model_path}")
            return None
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return None


# Modeli yükle
model = load_model()

# Görsel dönüşümlerini tanımla
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Sınıflandırma sonucunu görselleştirme
def visualize_prediction(probs, class_names):
    top_k = 5
    if len(class_names) < top_k:
        top_k = len(class_names)

    # En yüksek olasılıklı sınıfları al
    top_probs, top_indices = torch.topk(probs, k=top_k)

    # Numpy dizisine dönüştür
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    # Sınıf isimlerini al
    top_classes = [class_names[i] for i in top_indices]

    # Grafiği oluştur
    fig, ax = plt.subplots(figsize=(10, 6))

    # Yatay bar plot
    y_pos = np.arange(len(top_classes))
    ax.barh(y_pos, top_probs * 100, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_classes)
    ax.invert_yaxis()  # Etiketleri yukarıdan aşağıya sırala
    ax.set_xlabel('Olasılık (%)')
    ax.set_title('En Olası Sınıflar')

    # Değerleri bar'ların yanında göster
    for i, v in enumerate(top_probs):
        ax.text(v * 100 + 1, i, f"{v * 100:.1f}%", va='center')

    plt.tight_layout()

    # Geçici dosya olarak kaydet
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name)
        plt.close()
        return tmp.name


# Tahmin fonksiyonu
def predict(image):
    if model is None:
        return None, "Model yüklenemedi. Lütfen sistem yöneticinizle iletişime geçin.", None

    if image is None:
        return None, "Lütfen bir görsel yükleyin.", None

    try:
        # Görüntü ön işleme
        img_processed = transform(image).unsqueeze(0).to(device)

        # Tahmin yapma
        with torch.no_grad():
            outputs = model(img_processed)
            probs = F.softmax(outputs, dim=1)[0]

        # En yüksek olasılığa sahip sınıfı bul
        confidence, predicted_idx = torch.max(probs, 0)
        predicted_class = classes[predicted_idx.item()]
        confidence_score = confidence.item() * 100

        # Sonucu düzenle
        result_text = f"Tahmin: {predicted_class}\nGüven Skoru: {confidence_score:.2f}%"

        # Sınıflandırma sonucunu görselleştir
        viz_path = visualize_prediction(probs, classes)

        return image, result_text, viz_path
    except Exception as e:
        import traceback
        error_msg = f"Tahmin sırasında hata oluştu: {str(e)}"
        traceback.print_exc()
        return image, error_msg, None


# Örnek görüntü seçme fonksiyonu
def get_example_images():
    examples = []
    try:
        example_dir = "./raw-img"
        if not os.path.exists(example_dir):
            return []

        for class_name in classes:
            class_path = os.path.join(example_dir, class_name)
            if os.path.isdir(class_path):
                files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    # Her sınıftan bir örnek seç
                    examples.append(os.path.join(class_path, files[0]))
    except Exception as e:
        print(f"Örnek görseller yüklenirken hata oluştu: {e}")

    return examples


# Gradio arayüzü
def create_interface():
    with gr.Blocks(title="Görüntü Sınıflandırıcı") as interface:
        gr.Markdown("# 🔍 Görüntü Sınıflandırıcı")
        gr.Markdown("Bir görüntü yükleyin ve modelin hangi sınıfa ait olduğunu tahmin etmesini izleyin.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Görüntü Yükle")
                predict_btn = gr.Button("Tahmin Et", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(type="pil", label="Yüklenen Görüntü")
                output_text = gr.Textbox(label="Tahmin Sonucu")
                output_plot = gr.Image(label="Sınıf Olasılıkları")

        predict_btn.click(fn=predict, inputs=input_image, outputs=[output_image, output_text, output_plot])

        # Örnek görselleri ekle
        examples = get_example_images()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=input_image,
                outputs=[output_image, output_text, output_plot],
                fn=predict,
                cache_examples=True
            )

        gr.Markdown("## Nasıl Kullanılır")
        gr.Markdown("""
        1. 'Görüntü Yükle' alanına bir görüntü yükleyin veya örnek görsellerden birini seçin
        2. 'Tahmin Et' butonuna tıklayın
        3. Model tahminini ve güven skorunu sağ tarafta göreceksiniz
        """)

        if classes != ["Sınıf bilgisi yüklenemedi"]:
            class_info = ", ".join(classes)
            gr.Markdown(f"**Tanımlı Sınıflar:** {class_info}")

    return interface


# Ana fonksiyon
def main():
    interface = create_interface()
    interface.launch(share=False, server_name="127.0.0.1")


if __name__ == "__main__":
    main()