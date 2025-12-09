# 🎤 End-to-End Speech Emotion Recognition with Gender Information

This repository implements an **end-to-end deep learning model for Speech Emotion Recognition (SER)** using **raw audio waveforms** enhanced with **gender information** as an auxiliary feature.  
Unlike feature-engineered systems that rely on MFCC, pitch & spectral features, this model directly learns emotional features from raw waveform using deep neural networks.

---

## 🔥 Key Concept

| Feature | Explanation |
|--------|-------------|
| End-to-End | Model takes raw audio input directly — no MFCC feature engineering needed |
| Gender-Aware | Gender input improves emotion detection accuracy |
| Deep Learning | CNN-based architecture extracts emotion-specific patterns from waveform |
| Multi-Language | Works with Mandarin, English, German speech datasets |

---

## 🧠 Why Gender Information Matters?

Studies show that the **fundamental frequency, formants & emotional tone differ across male and female voices**.  
Integrating gender increases recognition accuracy and robustness across datasets.


---



## 🧰 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical contents:

```
numpy  
torch or tensorflow
librosa  
soundfile  
pandas  
tqdm  
scipy
```

---



## 📊 Model Flow

```
Raw Audio  ──▶  Waveform CNN Encoder ──▶ Feature Extraction ──┐
                      Gender Input ────────────────▶ Join ──▶ Emotion Classifier
```

Outputs emotions like:

```
😡 Angry
😊 Happy
😢 Sad
😐 Neutral
🤩 Excited
```

---

## 🔮 Future Improvements

- Real-time emotion classification (microphone streaming)
- Transformer-based architecture
- Multi-emotion multi-speaker support
- Emotion + Gender + Age + Accents multi-feature fusion

---


## 🤝 Contributing

Pull requests are welcome.  
Feel free to improve the model architecture, preprocessing pipeline or add deployment support.

---

## ⭐ Support

If you find this helpful, **star ⭐ the repository** to support the project.

