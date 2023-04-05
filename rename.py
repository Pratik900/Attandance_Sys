import os

# Set the path to the folder you want to rename
folder_path = "data\Train\Yash"

# Set the starting number for the new sequence
new_start_num = 1
# Loop through the files in the folder
for i, filename in enumerate(os.listdir(folder_path)):
    # Generate the new filename by adding the new sequence number
    new_filename = f"Yash.3.{new_start_num + i}.jpg"

    
    # Rename the file
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))