# New number to insert
import os

category = {
    'cat': 0,
    'dog': 1,
    'drone': 2,
    'phone': 3
    }

tasks = {
    'test': 0,
    'train': 1,
    'valid': 2
    }

file_direc = 'virtualENV/step1'
files = os.listdir(file_direc)


for task in tasks:
    for cats in files:
        if cats in category:
            new_number = category[cats]
            print(new_number)
            dirs = file_direc + '/' + cats + '/' + task + '/labels'
            ans = os.listdir(str(dirs))
            num_files = len(ans)
            iterator = 0
            for file in range(len(ans)):
                file_path = os.path.join(dirs, ans[file])

                with open(file_path, 'r') as file:
                    content = file.readlines()

                modified_content = [line.replace('0', str(new_number), 1) for line in content]

                with open(file_path, 'w') as file:
                    file.writelines(modified_content)
    
            #print(iterator, cats, file_path)
        



#file_path = '0001_jpg.rf.2a58e3cb39bf91af5e1c77b37400dce7.txt'




# Replace all occurrences of '0' in the first column with '3'

# Write the modified content back to the file



print("First number replaced successfully.")
