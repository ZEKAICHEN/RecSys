from zipfile import ZipFile
import tarfile

fname = 'data/dac.tar.gz'
# with ZipFile(path, 'r') as zipObj:
#     zipObj.extractall('data/')

if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()
elif fname.endswith("tar"):
    tar = tarfile.open(fname, "r:")
    tar.extractall()
    tar.close()