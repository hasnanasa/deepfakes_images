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
import sys

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Matériel utilisé : {device}")

print("Chargement de CLIP (ViT-B/32)...")
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. Fonction d'extraction
def extraire_features(dossier, label, est_train=False):
    features, labels = [], []
    if not os.path.exists(dossier): 
        print(f"❌ Dossier introuvable : {dossier}")
        return np.array([]), np.array([])
    
    extensions_valides = ('.png', '.jpg', '.jpeg', '.webp', '.jfif')
    images = [f for f in os.listdir(dossier) if f.lower().endswith(extensions_valides)]
    
    if len(images) == 0:
        print(f"⚠️ Aucune image dans {dossier}")
        return np.array([]), np.array([])

    for nom_image in tqdm(images, desc=f"Extraction {'Train' if est_train else 'Test'}"):
        chemin = os.path.join(dossier, nom_image)
        try:
            img = Image.open(chemin).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feature = model.encode_image(img_tensor)
                feature /= feature.norm(dim=-1, keepdim=True) # Normalisation L2
                
            features.append(feature.cpu().numpy().flatten())
            labels.append(label)
        except Exception as e:
            pass
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    print("\n--- EXTRACTION DES DONNÉES (TRAIN) ---")
    X_tr_real, y_tr_real = extraire_features("data/train/real", 0, True)
    X_tr_fake, y_tr_fake = extraire_features("data/train/fake", 1, True)
    
    if len(X_tr_real) == 0 or len(X_tr_fake) == 0:
        print("❌ Erreur : Les dossiers Train sont vides. As-tu lancé le découpage ?")
        sys.exit()

    X_train = np.concatenate([X_tr_real, X_tr_fake])
    y_train = np.concatenate([y_tr_real, y_tr_fake])

    print("\n--- EXTRACTION DES DONNÉES (TEST) ---")
    X_te_real, y_te_real = extraire_features("data/test/real", 0, False)
    X_te_fake, y_te_fake = extraire_features("data/test/fake", 1, False)
    
    X_test = np.concatenate([X_te_real, X_te_fake])
    y_test = np.concatenate([y_te_real, y_te_fake])

    # 3. Normalisation (Crucial pour la Régression Logistique)
    print("\nNormalisation des caractéristiques...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Entraînement de la Baseline (Régression Logistique)
    print("\n--- ENTRAÎNEMENT (Régression Logistique avec C=0.01) ---")
    # C=0.01 empêche le modèle de mémoriser (overfitting)
    classifieur = LogisticRegression(max_iter=2000, C=0.01)
    classifieur.fit(X_train_scaled, y_train)

    # 5. Évaluation
    y_pred = classifieur.predict(X_test_scaled)
    y_proba = classifieur.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n✅ Précision globale (Accuracy) : {acc * 100:.2f}%")
    print(f"✅ Score AUC : {auc:.4f} (1.0 = parfait, 0.5 = hasard)\n")
    
    print("Rapport détaillé :")
    print(classification_report(y_test, y_pred, target_names=["Réelle (0)", "Fake (1)"]))

    # 6. Sauvegarde
    joblib.dump(scaler, "final_scaler.pkl")
    joblib.dump(classifieur, "final_lr.pkl")
    print("💾 Modèle et Scaler sauvegardés avec succès !")