import os
import yaml
import shutil

def check_and_fix_yolo_setup():
    """
    Vérifie et corrige la configuration YOLO
    """
    # Chemins
    base_dir = 'data'
    train_images = os.path.join(base_dir, 'train', 'images')
    valid_images = os.path.join(base_dir, 'valid', 'images')
    yaml_path = os.path.join(base_dir, 'config.yaml')

    # 1. Vérifier que les dossiers existent
    print("Vérification des dossiers...")
    for path in [train_images, valid_images]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Créé le dossier manquant: {path}")

    # 2. Vérifier qu'il y a des images dans les dossiers
    train_files = os.listdir(train_images)
    valid_files = os.listdir(valid_images)
    
    print(f"Images d'entraînement trouvées: {len(train_files)}")
    print(f"Images de validation trouvées: {len(valid_files)}")

    # Si le dossier de validation est vide mais qu'il y a des images d'entraînement,
    # copier quelques images d'entraînement vers la validation
    if len(valid_files) == 0 and len(train_files) > 0:
        print("Copie de quelques images d'entraînement vers la validation...")
        for file in train_files[:min(3, len(train_files))]:  # Copier jusqu'à 3 images
            src = os.path.join(train_images, file)
            dst = os.path.join(valid_images, file)
            shutil.copy2(src, dst)
            
            # Copier aussi les labels correspondants si ils existent
            label_src = os.path.join(base_dir, 'train', 'labels', file.replace('.jpg', '.txt'))
            label_dst = os.path.join(base_dir, 'valid', 'labels', file.replace('.jpg', '.txt'))
            if os.path.exists(label_src):
                os.makedirs(os.path.join(base_dir, 'valid', 'labels'), exist_ok=True)
                shutil.copy2(label_src, label_dst)

    # 3. Créer/Mettre à jour le fichier data.yaml
    yaml_content = {
        'path': './data',
        'train': 'train/images',
        'val': 'valid/images',
        'nc': 2,
        'names': ['puces', 'bulles']
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"Fichier config.yaml mis à jour: {yaml_path}")

    # 4. Vérifier les permissions
    print("\nVérification des permissions...")
    for path in [train_images, valid_images]:
        try:
            test_file = os.path.join(path, 'test_permissions.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"Permissions OK pour: {path}")
        except Exception as e:
            print(f"ATTENTION: Problème de permissions pour {path}: {str(e)}")

    return {
        'train_images': len(train_files),
        'valid_images': len(valid_files),
        'yaml_path': yaml_path
    }

if __name__ == "__main__":
    results = check_and_fix_yolo_setup()
    print("\nRésumé de la configuration:")
    print(f"- Images d'entraînement: {results['train_images']}")
    print(f"- Images de validation: {results['valid_images']}")
    print(f"- Fichier de configuration: {results['yaml_path']}")