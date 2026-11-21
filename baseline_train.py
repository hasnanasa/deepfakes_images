import os
import torch
import clip
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import joblib

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extraire_features(dossier, label, est_train=False):
    features, labels = [], []
    if not os.path.exists(dossier): return np.array([]), np.array([])
    
    images = [f for f in os.listdir(dossier) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for nom_image in tqdm(images, desc=f"Extraction {'Train' if est_train else 'Test'}"):
        chemin = os.path.join(dossier, nom_image)
        try:
            img = Image.open(chemin).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feature = model.encode_image(img_tensor)
                feature /= feature.norm(dim=-1, keepdim=True)
                
            features.append(feature.cpu().numpy().flatten())
            labels.append(label)
        except Exception as e:
            pass
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    print("\n--- EXTRACTION ---")
    X_tr_real, y_tr_real = extraire_features("data/train/real", 0, True)
    X_tr_fake, y_tr_fake = extraire_features("data/train/fake", 1, True)
    X_train = np.concatenate([X_tr_real, X_tr_fake])
    y_train = np.concatenate([y_tr_real, y_tr_fake])

    X_te_real, y_te_real = extraire_features("data/test/real", 0, False)
    X_te_fake, y_te_fake = extraire_features("data/test/fake", 1, False)
    X_test = np.concatenate([X_te_real, X_te_fake])
    y_test = np.concatenate([y_te_real, y_te_fake])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- ENTRAÎNEMENT (Régularisation Forte C=0.01) ---")
    # C=0.01 pénalise fortement le modèle s'il essaie de mémoriser
    classifieur = LogisticRegression(max_iter=2000, C=0.01) 
    classifieur.fit(X_train_scaled, y_train)

    # Évaluation
    y_pred = classifieur.predict(X_test_scaled)
    y_proba = classifieur.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n✅ Précision (Accuracy) : {acc * 100:.2f}%")
    print(f"✅ Score AUC : {auc:.4f} (1.0 = parfait, 0.5 = hasard)\n")
    
    print("Exemples de probabilités sur le Test (pour voir si le modèle est trop confiant) :")
    for i in range(5):
        print(f" - Image {i+1} : Proba Fake = {y_proba[i]*100:.2f}%")

    joblib.dump(scaler, "baseline_scaler.pkl")
    joblib.dump(classifieur, "baseline_lr.pkl")