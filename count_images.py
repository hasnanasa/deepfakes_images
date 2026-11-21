import os

def compter_images(dossier):
    if not os.path.exists(dossier):
        return None
    
    # Extensions d'images valides
    extensions_valides = ('.png', '.jpg', '.jpeg', '.webp', '.jfif', '.bmp')
    
    # Compter uniquement les fichiers avec une extension valide
    fichiers = [f for f in os.listdir(dossier) 
                if os.path.isfile(os.path.join(dossier, f)) and f.lower().endswith(extensions_valides)]
    
    return len(fichiers)

# --- CHEMINS DES DOSSIERS ---
# On gère le cas où le dossier s'appelle "fake" ou "fakes"
dossier_real = "data/images/real"
dossier_fake = "data/images/fake"
if not os.path.exists(dossier_fake) and os.path.exists("data/images/fakes"):
    dossier_fake = "data/images/fakes"

print("🔍 Vérification du Dataset...\n" + "="*30)

nb_real = compter_images(dossier_real)
nb_fake = compter_images(dossier_fake)

if nb_real is None:
    print(f"❌ Dossier introuvable : {dossier_real}")
else:
    print(f"✅ Images RÉELLES (Real) : {nb_real}")

if nb_fake is None:
    print(f"❌ Dossier introuvable : {dossier_fake}")
else:
    print(f"✅ Images FAUSSES (Fake) : {nb_fake}")

print("="*30)

if nb_real and nb_fake:
    total = nb_real + nb_fake
    print(f"📊 Total d'images : {total}")
    
    # Vérifier l'équilibre
    ratio = nb_real / nb_fake if nb_fake > 0 else 0
    print(f"⚖️ Ratio (Real / Fake) : {ratio:.2f}")
    
    if 0.8 <= ratio <= 1.2:
        print("🎉 Excellent ! Ton dataset est bien équilibré.")
    else:
        print("⚠️ Attention : Ton dataset est déséquilibré. Ajoute des images à la classe la plus faible.")