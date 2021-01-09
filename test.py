import scipy.io as scio

# dataFile = '/Users/Konyaka/Downloads/imagelabels.mat'
dataFile = '/Users/Konyaka/Downloads/setid.mat'
data = scio.loadmat(dataFile)
print(data)
