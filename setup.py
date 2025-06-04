import os

def create_sign_language_folders():
    """
    Create folders for 7 sign language words in the existing dataset folder.
    """
    # List of 7 common sign language words
    words = ["Again","Bathroom","Eat","Hello","Help","How are you","No","Please","Sorry","thanks","What","yes" ]
    
    # Main dataset folder path
    dataset_folder = "dataset"
    
    # Check if dataset folder exists
    if not os.path.exists(dataset_folder):
        print(f"Error: '{dataset_folder}' folder not found. Creating it.")
        os.makedirs(dataset_folder)
    else:
        print(f"Using existing '{dataset_folder}' folder.")
    
    # Create a folder for each word if it doesn't exist
    for word in words:
        word_folder = os.path.join(dataset_folder, word)
        if not os.path.exists(word_folder):
            os.makedirs(word_folder)
            print(f"Created folder: {word_folder}")
        else:
            print(f"Folder already exists: {word_folder}")
    
    print(f"\nSuccessfully created folders for {len(words)} sign language words.")
    print("Words:", ", ".join(words))

if __name__ == "__main__":
    create_sign_language_folders()