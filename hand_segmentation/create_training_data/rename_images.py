import os

# path = 'C:/Users/ralf-/OneDrive/!Uni/Seminar/data/bla'
path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/03_resized_images/test_set/persons/p1'
files = os.listdir(path)

for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index+83), '.jpg'])))

