import h5py

file = 'f30k_resnet152_temp.h5'
ds = h5py.File(file, 'r')

print(len(ds['images']['flickr30k_images'].keys()))
