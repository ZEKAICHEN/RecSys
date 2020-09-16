from zipfile import ZipFile

path = 'data/ml-25m.zip'
with ZipFile(path, 'r') as zipObj:
    zipObj.extractall('data/')