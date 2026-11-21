import os
import random
import shutil

# --- CHEMINS SOURCES ---
source_real = "data/images/real"
source_fake = "data/images/fake"

# --- CHEMINS DE DESTINATION ---
train_real = "data/train/real"
train_fake = "data/train/fake"
test_real = "data/test/real"
test_fake = "data/test/fake"

# 1. Nettoyage des anciens dossiers Train et Test (pour repartir de zéro)
print("🧹 Nettoyage des anciens dossiers...")
for dossier in ["data/train", "data/test"]:
    if os.path.exists(dossier):
        shutil.rmtree(dossier)

# 2. Création de l'arborescence
for dossier in [train_real, train_fake, test_real, test_fake]:
    os.makedirs(dossier, exist_ok=True)

# 3. Fonction de découpage
def split_data(source, dest_train, dest_test, nom_classe):
    if not os.path.exists(source):
        print(f"❌ Dossier introuvable : {source}")
        return

    # Récupérer toutes les images (.jpg car on a standardisé)
    images = [f for f in os.listdir(source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Mélanger aléatoirement avec une seed fixe pour la reproductibilité
    random.seed(42)
    random.shuffle(images)
    
    # Calculer l'index pour 80%
    index_split = int(len(images) * 0.8)
    
    images_train = images[:index_split]
    images_test = images[index_split:]
    
    # Copier dans Train
    for img in images_train:
        shutil.copy(os.path.join(source, img), os.path.join(dest_train, img))
        
    # Copier dans Test
    for img in images_test:
        shutil.copy(os.path.join(source, img), os.path.join(dest_test, img))
        
    print(f"✅ {nom_classe} : {len(images_train)} dans Train, {len(images_test)} dans Test.")

if __name__ == "__main__":
    print("✂️ Création des dossiers Train et Test (80% / 20%)...\n")
    split_data(source_real, train_real, test_real, "RÉELLES")
    split_data(source_fake, train_fake, test_fake, "FAKES")
    print("\n🎉 Terminé ! Tu peux maintenant lancer l'entraînement.")