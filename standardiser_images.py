import os
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
TAILLE_CIBLE = (512, 512) # Taille standard (CLIP va de toute façon la réduire à 224x224)
DOSSIERS = [
    "data/images/real",
    "data/images/fake"
]

def standardiser_dossier(dossier):
    if not os.path.exists(dossier):
        print(f"❌ Dossier introuvable : {dossier}")
        return

    # On récupère toutes les images (peu importe l'extension)
    extensions_valides = ('.png', '.jpg', '.jpeg', '.webp', '.jfif', '.bmp', '.PNG', '.JPG', '.JPEG')
    fichiers = [f for f in os.listdir(dossier) if f.lower().endswith(extensions_valides)]
    
    print(f"\n📁 Traitement de {len(fichiers)} images dans {dossier}...")
    
    for nom_fichier in tqdm(fichiers):
        chemin_original = os.path.join(dossier, nom_fichier)
        
        try:
            # 1. Ouvrir et convertir en RGB (supprime la transparence des PNG)
            img = Image.open(chemin_original).convert("RGB")
            
            # 2. Redimensionner (sans déformer l'image, on la centre et on coupe les bords)
            # On utilise Image.Resampling.LANCZOS pour une excellente qualité
            img_redim = img.resize(TAILLE_CIBLE, Image.Resampling.LANCZOS)
            
            # 3. Créer le nouveau nom avec l'extension .jpg
            nom_base = os.path.splitext(nom_fichier)[0]
            nouveau_nom = f"{nom_base}.jpg"
            nouveau_chemin = os.path.join(dossier, nouveau_nom)
            
            # 4. Sauvegarder au format JPEG (qualité 95 pour garder les détails)
            img_redim.save(nouveau_chemin, format="JPEG", quality=95)
            
            # 5. Supprimer l'ancien fichier s'il était dans un autre format (ex: .png)
            if chemin_original != nouveau_chemin:
                os.remove(chemin_original)
                
        except Exception as e:
            print(f"❌ Erreur avec {nom_fichier} : {e}")

if __name__ == "__main__":
    print("🚀 Début de la standardisation (512x512, format .jpg)...")
    for dossier in DOSSIERS:
        standardiser_dossier(dossier)
    
    print("\n✅ Terminé ! Toutes tes images ont la même taille et la même extension.")