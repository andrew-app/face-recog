import os

path = 'F:\\OpenCV\\FaceLM\\rename_img\\images\\'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path): #Generate the file names in a directory tree by walking the tree either top-down or bottom-up
    for file in f: #read files that are in folder
        print(file)
        if '.jpg' in file: #only care about jpg
            files.append(os.path.join(r, file))
print(files)

_ext = ".jpg"
_src = "F:\\OpenCV\\FaceLM\\rename_img\\images\\"

k = 3

for i in range(0,2):
    for j in range(4,5):
        os.rename(files[i], _src+'Keanu' + str(k)+_ext)
        k = j
        break





