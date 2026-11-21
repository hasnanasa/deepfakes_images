import os
import random
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

DOSSIER_FAKE = "data/images/fake"
NOMBRE_IMAGES = 40
TAILLE_CIBLE = (512, 512)

def appliquer_traitement_et_ecraser(chemin_image):
    try:
        # 1. Lire l'image
        img_cv = cv2.imread(chemin_image)
        if img_cv is None: return False
        
        # 2. Traitement Samsung/Oppo (Sur-accentuation)
        kernel_sharpening = np.array([[-1,-1,-1], 
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
        img_cv = cv2.filter2D(img_cv, -1, kernel_sharpening)
        
        # 3. Conversion pour les filtres de couleur
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # 4. Lissage de peau léger
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # 5. Saturation et contraste (Look téléphone)
        img_pil = ImageEnhance.Color(img_pil).enhance(1.15)
        img_pil = ImageEnhance.Contrast(img_pil).enhance(1.05)
        
        # 6. Redimensionner à 512x512 (standardisation)
        img_pil = img_pil.resize(TAILLE_CIBLE, Image.Resampling.LANCZOS)
        
        # 7. ÉCRASER l'image originale (même nom, même chemin)
        img_pil.save(chemin_image, format="JPEG", quality=78)
        return True
    except Exception as e:
        print(f"Erreur avec {chemin_image} : {e}")
        return False

if __name__ == "__main__":
    print("📱 Application du traitement Samsung/Oppo sur 40 images Fake (Écrasement)...")
    
    # On récupère les images
    images = [f for f in os.listdir(DOSSIER_FAKE) if f.lower().endswith('.jpg')]
    
    if len(images) < NOMBRE_IMAGES:
        print("⚠️ Pas assez d'images dans le dossier.")
    else:
        # On sélectionne 40 images au hasard
        images_a_modifier = random.sample(images, NOMBRE_IMAGES)
        
        for nom_image in tqdm(images_a_modifier):
            chemin = os.path.join(DOSSIER_FAKE, nom_image)
            appliquer_traitement_et_ecraser(chemin)

    print("\n✅ Terminé ! 40 images ont été écrasées avec le look Samsung/Oppo en 512x512.")
    print("👉 Tu n'as plus qu'à lancer ton split (split_grouped.py ou split_train_test.py) puis l'entraînement.")