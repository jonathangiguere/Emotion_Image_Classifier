import os
import shutil

# Part 1 of 2
# Had to split it up because doing all decompressed data at once broke the VM

# used shell to manually move the decompressed data to this folder
current_dir = '/home/ubuntu/capstone/decompressed_data'
# Specify folder where all images will live
dest_dir = '/home/ubuntu/capstone/all_images'
counter = 0

for subdir, dirs, files in os.walk(current_dir):
    #print(files)
    for file in files:
        full_path = os.path.join(subdir, file)
        shutil.copy(full_path, dest_dir)
        counter = counter + 1
print(counter)

