import os
import shutil

if __name__ == "__main__":

    path = input("Which one are you restructuring (train or test folder) ? ")

    try:
        os.chdir(path)
    except FileNotFoundError:
        print("Your specified directory wasn't found.")

    img_types = ['1mm B30', '1mm D45', '3mm B30', '3mm D45']

    for img_type in img_types:
        folders = os.listdir(img_type)
        folders = [folder for folder in folders if not folder.endswith(".zip") and folder != '.DS_Store']

        for folder in folders:
            patient_path = '/'.join([img_type, folder])
            patients = [patient for patient in os.listdir(patient_path) if patient != '.DS_Store']

            for patient in patients:
                folders_to_rm_path = '/'.join([patient_path, patient])
                folders_to_rm = [folder_to_rm for folder_to_rm in os.listdir(folders_to_rm_path)
                                 if folder_to_rm != '.DS_Store' and not folder_to_rm.endswith('.IMA')]

                for folder_to_rm in folders_to_rm:
                    items_path = '/'.join([folders_to_rm_path, folder_to_rm])
                    items = [item for item in os.listdir(items_path) if item.endswith('.IMA')]
                    for item in items:
                        shutil.move(f"{img_type}/{folder}/{patient}/{folder_to_rm}/{item}",
                                    f"{img_type}/{folder}/{patient}")
                    os.rmdir(f"{img_type}/{folder}/{patient}/{folder_to_rm}")
