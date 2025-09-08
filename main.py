# Instalar dependências
# !pip install tensorflow matplotlib opencv-python pillow

import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from PIL import Image

# -----------------------------
# Função para construir SRCNN simplificado
# -----------------------------
def build_simple_srcnn():
    """SRCNN simplificado com inicialização melhor"""
    model = Sequential([
        Conv2D(64, (9,9), activation='relu', padding='same', 
               input_shape=(None, None, 1),
               kernel_initializer='he_normal'),
        Conv2D(32, (1,1), activation='relu', padding='same',
               kernel_initializer='he_normal'),
        Conv2D(1, (5,5), activation='sigmoid', padding='same',  # Sigmoid para garantir [0,1]
               kernel_initializer='glorot_normal')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------------
# Implementação de filtros tradicionais para comparação
# -----------------------------
def apply_sharpening_filter(img):
    """Aplicar filtro de nitidez"""
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return np.clip(sharpened, 0, 255)

def apply_unsharp_mask(img, sigma=1.0, strength=1.5):
    """Aplicar máscara de nitidez (unsharp mask)"""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    unsharp = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    return np.clip(unsharp, 0, 255)

def lanczos_upscale(img, scale_factor):
    """Upscaling com Lanczos (melhor que bicúbico)"""
    h, w = img.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

# -----------------------------
# Função para preprocessar imagem
# -----------------------------
def create_test_image():
    """Criar uma imagem de teste mais complexa"""
    img = np.zeros((200, 200), dtype=np.uint8)
    
    # Fundo gradiente
    for i in range(200):
        img[i, :] = int(50 + (i/200) * 100)
    
    # Adicionar formas geométricas
    cv2.rectangle(img, (50, 50), (150, 150), 255, 2)
    cv2.circle(img, (100, 100), 30, 200, 2)
    cv2.circle(img, (100, 100), 15, 150, -1)
    
    # Linhas diagonais
    cv2.line(img, (20, 20), (180, 180), 180, 2)
    cv2.line(img, (20, 180), (180, 20), 180, 2)
    
    # Texto com diferentes tamanhos
    cv2.putText(img, 'TEST', (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(img, 'SR', (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 220, 1)
    
    # Adicionar ruído controlado
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def load_image():
    """Tentar carregar imagem ou usar teste"""
    # Headers para evitar bloqueio de sites
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",      
    ]
    
    for url in urls:
        try:
            print(f"Tentando: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img = np.array(img)
            
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Redimensionar para tamanho padrão
            img = cv2.resize(img, (200, 200))
            print("✓ Imagem carregada com sucesso!")
            return img
            
        except Exception as e:
            print(f"✗ Falhou: {e}")
            continue
    
    print("Usando imagem de teste...")
    return create_test_image()

# -----------------------------
# Função principal melhorada
# -----------------------------
def main():
    # Carregar imagem
    img_original = load_image()
    print(f"Imagem carregada: {img_original.shape}")
    
    # Parâmetros
    scale_factor = 2
    h, w = img_original.shape
    low_h, low_w = h // scale_factor, w // scale_factor
    
    # Simular baixa resolução
    img_low = cv2.resize(img_original, (low_w, low_h), interpolation=cv2.INTER_AREA)
    
    # Métodos de upscaling
    bicubic = cv2.resize(img_low, (w, h), interpolation=cv2.INTER_CUBIC)
    lanczos = lanczos_upscale(img_low, scale_factor)
    
    # Aplicar filtros de pós-processamento
    sharpened = apply_sharpening_filter(bicubic.astype(np.float32)).astype(np.uint8)
    unsharp = apply_unsharp_mask(bicubic.astype(np.float32)).astype(np.uint8)
    
    # SRCNN (demonstrativo - pesos não treinados)
    print("Aplicando SRCNN...")
    srcnn = build_simple_srcnn()
    
    # Preparar entrada
    input_tensor = bicubic.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(np.expand_dims(input_tensor, 0), -1)
    
    # Predição
    pred = srcnn.predict(input_tensor, verbose=0)
    srcnn_result = (pred[0, :, :, 0] * 255).astype(np.uint8)
    
    # -----------------------------
    # Visualização comparativa
    # -----------------------------
    plt.figure(figsize=(18, 12))
    
    methods = [
        ("Original", img_original),
        (f"Baixa Res. ({low_w}×{low_h})", img_low),
        ("Bicúbica", bicubic),
        ("Lanczos", lanczos),
        ("Bicúbica + Sharpen", sharpened),
        ("Bicúbica + Unsharp", unsharp),
        ("SRCNN (não treinado)", srcnn_result)
    ]
    
    for i, (title, img) in enumerate(methods):
        plt.subplot(2, 4, i+1)
        plt.title(title, fontsize=10)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # -----------------------------
    # Métricas de qualidade
    # -----------------------------
    def calculate_psnr(img1, img2):
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def calculate_ssim_simple(img1, img2):
        """SSIM simplificado"""
        mu1, mu2 = img1.mean(), img2.mean()
        sigma1, sigma2 = img1.var(), img2.var()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        
        return max(0, min(1, ssim))
    
    # Calcular métricas incluindo referências
    results = []
    
    # REFERÊNCIA: Imagem de baixa resolução (pior caso)
    psnr_low = calculate_psnr(img_original, cv2.resize(img_low, (w, h), cv2.INTER_NEAREST))
    ssim_low = calculate_ssim_simple(img_original.astype(float), 
                                   cv2.resize(img_low, (w, h), cv2.INTER_NEAREST).astype(float))
    results.append(("🔻 Baixa Res. (referência)", psnr_low, ssim_low))
    
    # Métodos de super-resolução
    for name, img in [("Bicúbica", bicubic), 
                     ("Lanczos", lanczos),
                     ("Bicúbica + Sharpen", sharpened),
                     ("Bicúbica + Unsharp", unsharp),
                     ("SRCNN (não treinado)", srcnn_result)]:
        psnr = calculate_psnr(img_original, img)
        ssim = calculate_ssim_simple(img_original.astype(float), img.astype(float))
        results.append((name, psnr, ssim))
    
    # TETO TEÓRICO: Original vs Original
    results.append(("🎯 Original (teto teórico)", float('inf'), 1.0))
    
    print("\n" + "="*65)
    print("COMPARAÇÃO DE MÉTODOS DE SUPER-RESOLUÇÃO")
    print("="*65)
    print(f"{'Método':<25} {'PSNR (dB)':<12} {'SSIM':<8} {'Status':<12}")
    print("-"*65)
    
    for i, (name, psnr, ssim) in enumerate(results):
        # Formatação especial para infinito
        psnr_str = "∞" if psnr == float('inf') else f"{psnr:8.2f}"
        
        # Classificação de qualidade
        if psnr == float('inf'):
            status = "PERFEITO"
        elif psnr >= 35:
            status = "EXCELENTE"
        elif psnr >= 30:
            status = "BOM"
        elif psnr >= 25:
            status = "ACEITÁVEL"
        else:
            status = "RUIM"
            
        print(f"{name:<25} {psnr_str:>8}    {ssim:>6.3f}   {status:<12}")
    
    # Análise adicional
    method_results = results[1:-1]  # Excluir referências
    best_psnr = max(method_results, key=lambda x: x[1])
    best_ssim = max(method_results, key=lambda x: x[2])
    
    print("\n📊 ANÁLISE DETALHADA:")
    print("-" * 40)
    
    # Calcular ganhos
    baseline_psnr = results[0][1]  # Baixa resolução
    baseline_ssim = results[0][2]
    
    for name, psnr, ssim in method_results:
        gain_psnr = psnr - baseline_psnr
        gain_ssim = ssim - baseline_ssim
        print(f"{name}:")
        print(f"  └ Ganho PSNR: {gain_psnr:+.2f} dB")
        print(f"  └ Ganho SSIM: {gain_ssim:+.3f}")
        print()
    
    print("📝 GUIA DE INTERPRETAÇÃO:")
    print("• PSNR > 35 dB: Excelente qualidade")
    print("• PSNR 30-35 dB: Boa qualidade") 
    print("• PSNR 25-30 dB: Qualidade aceitável")
    print("• PSNR < 25 dB: Qualidade ruim")
    print()
    print("• SSIM > 0.9: Excelente similaridade")
    print("• SSIM 0.8-0.9: Boa similaridade")
    print("• SSIM 0.7-0.8: Similaridade aceitável") 
    print("• SSIM < 0.7: Similaridade ruim")
    
    print(f"\n🏆 MELHORES RESULTADOS:")
    print(f"• Maior PSNR: {best_psnr[0]} ({best_psnr[1]:.2f} dB)")
    print(f"• Maior SSIM: {best_ssim[0]} ({best_ssim[2]:.3f})")
    
    # Recomendação prática
    if best_psnr[0] == best_ssim[0]:
        print(f"• 🎯 Recomendado: {best_psnr[0]} (melhor em ambas métricas)")
    else:
        print(f"• 🤔 Trade-off: {best_psnr[0]} (PSNR) vs {best_ssim[0]} (SSIM)")

# -----------------------------
# Executar
# -----------------------------
if __name__ == "__main__":
    main()
