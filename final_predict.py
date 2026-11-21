import torch
import clip
from PIL import Image
import sys, os
import numpy as np
import joblib

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Chargement de CLIP et du modèle de détection...")

model, preprocess = clip.load("ViT-B/32", device=device)

try:
    scaler = joblib.load("final_scaler.pkl")
    classifieur = joblib.load("final_lr.pkl")
except FileNotFoundError:
    print("❌ Erreur : Impossible de trouver les fichiers du modèle. Lance final_train.py d'abord.")
    sys.exit()

def predire_image(chemin_image):
    if not os.path.exists(chemin_image):
        print(f"❌ L'image {chemin_image} n'existe pas.")
        return

    try:
        img = Image.open(chemin_image).convert('RGB')
    except Exception as e:
        print(f"❌ Impossible de lire l'image : {e}")
        return

    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feature = model.encode_image(img_tensor)
        feature /= feature.norm(dim=-1, keepdim=True)
    
    feature_np = scaler.transform(feature.cpu().numpy())
    
    prediction = classifieur.predict(feature_np)[0]
    proba = classifieur.predict_proba(feature_np)[0]
    
    print("\n" + "="*40)
    print(f"📸 Image analysée : {os.path.basename(chemin_image)}")
    if prediction == 1:
        print(f"🚨 VERDICT : **FAKE (Générée par IA)**")
        print(f"Confiance : {proba[1]*100:.2f}%")
    else:
        print(f"✅ VERDICT : **RÉELLE (Authentique)**")
        print(f"Confiance : {proba[0]*100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python final_predict.py <chemin_vers_image>")
    else:
        # On récupère le chemin passé dans le terminal
        chemin = sys.argv[1]
        predire_image(chemin)