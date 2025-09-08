# 🖼️ Super-Resolução de Imagens (SRCNN & Filtros Tradicionais)

Repositório: [sr](https://github.com/vitor-souza-ime/sr)  
Arquivo principal: `main.py`

Este projeto demonstra diferentes abordagens de **super-resolução de imagens**, comparando técnicas clássicas de interpolação e filtragem com uma versão simplificada da **SRCNN (Super-Resolution Convolutional Neural Network)**.

---

## 📌 Funcionalidades

- **Geração/Carregamento de imagem de teste**  
  - Carrega uma imagem de exemplo da web ou gera uma imagem sintética (formas geométricas, texto e ruído controlado).  

- **Métodos de super-resolução comparados:**
  - Redimensionamento **bicúbico**
  - Redimensionamento **Lanczos**
  - Bicúbico + **Sharpening**
  - Bicúbico + **Unsharp Mask**
  - **SRCNN simplificado** (rede neural convolucional – não treinada neste exemplo)

- **Métricas de avaliação:**
  - **PSNR (Peak Signal-to-Noise Ratio)**
  - **SSIM (Structural Similarity Index – simplificado)**

- **Visualização comparativa** em grade de imagens com Matplotlib.

---

## 📦 Dependências

Instale as bibliotecas necessárias com:

```bash
pip install tensorflow matplotlib opencv-python pillow requests
````

---

## ▶️ Como executar

Clone o repositório e rode o script principal:

```bash
git clone https://github.com/vitor-souza-ime/sr.git
cd sr
python main.py
```

---

## 📊 Saída esperada

1. **Visualização comparativa** (abrirá uma janela com as imagens):

   * Original
   * Baixa resolução
   * Bicúbica
   * Lanczos
   * Bicúbica + Sharpen
   * Bicúbica + Unsharp
   * SRCNN (não treinado)

2. **Tabela de métricas de qualidade (PSNR e SSIM):**

```
=================================================================
COMPARAÇÃO DE MÉTODOS DE SUPER-RESOLUÇÃO
=================================================================
Método                    PSNR (dB)    SSIM     Status
-----------------------------------------------------------------
🔻 Baixa Res. (referência)   22.10      0.650   RUIM       
Bicúbica                     28.35      0.812   ACEITÁVEL
Lanczos                      29.10      0.825   ACEITÁVEL
Bicúbica + Sharpen           27.80      0.800   ACEITÁVEL
Bicúbica + Unsharp           28.05      0.815   ACEITÁVEL
SRCNN (não treinado)         24.50      0.700   RUIM       
🎯 Original (teto teórico)    ∞          1.000   PERFEITO   
```

*(Valores de exemplo, variam conforme a imagem carregada e ruído adicionado.)*

3. **Análise automática de ganhos e recomendação prática** (qual método se destacou em PSNR e SSIM).

---

## 📖 Estrutura do código

* `build_simple_srcnn()` → Define a CNN simplificada de super-resolução.
* `apply_sharpening_filter()` → Aplica filtro de nitidez clássico.
* `apply_unsharp_mask()` → Aplica máscara de nitidez (unsharp).
* `lanczos_upscale()` → Redimensionamento com interpolação Lanczos.
* `create_test_image()` → Gera uma imagem sintética de teste.
* `load_image()` → Tenta carregar imagem externa ou usa a de teste.
* `main()` → Função principal: executa comparações, mostra imagens e calcula métricas.

---

## ⚠️ Observação

* A **SRCNN aqui não está treinada** – serve apenas para demonstrar a arquitetura.
* Para obter ganhos reais de super-resolução, seria necessário treinar a rede em um conjunto de dados de alta e baixa resolução.

---

