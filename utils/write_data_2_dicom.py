"""
Write Numpy to Dicom
"""
import SimpleITK as sitk
import numpy as np
import csv
import pandas as pd
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import dicom
import dicom.UID
import dicom_numpy
import datetime, time

from dicom.dataset import Dataset, FileDataset
from glob import glob



def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)


luna_path = './'
lun_subset_path = luna_path + 'train/'
file_list = glob(lun_subset_path + '*.mhd')
output_path = luna_path + 'npy/'
working_path = luna_path + 'output/'
if os.path.isdir(luna_path + '/npy'):
    pass
else:
    os.mkdir(luna_path + '/npy')

if os.path.isdir(luna_path + '/output'):
    pass
else:
    os.mkdir(luna_path + '/output')

df_node = pd.read_csv(luna_path + 'csv/train/' + 'annotations.csv')
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna()


def _read_file_to_numpy(img_file):
    mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
    if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
        print('filename = %s,slice = %d,height =  %d, width = %d, type = %s' % (img_file, num_z, height, width,img_array.dtype))
        return img_array, True


def _write_numpy_2_dicom(pixel_array, dcmfilename):
    filename = dcmfilename[dcmfilename.index('-')+1:dcmfilename.index('.mhd')]
    print(filename)
    for i in range(pixel_array.shape[0]):
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"  # !! Need valid UID here for real work
        file_meta.ImplementationClassUID = "1.2.3.4"  # !!! Need valid UIDs here

        ds = FileDataset(dcmfilename, {},file_meta=file_meta, preamble="\0"*128)
        ds.Modality = 'WSD'
        ds.ContentDate = str(datetime.date.today()).replace('-', '')
        ds.ContentTime = str(time.time())  # milliseconds since the epoch
        ds.StudyInstanceUID = '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
        ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
        ds.SOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        ds.SOPClassUID = 'Secondary Capture Image Storage'
        ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'

        ## These are the necessary imaging components of the FileDataset object.
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SmallestImagePixelValue = '\\x00\\x00'
        ds.LargestImagePixelValue = '\\xff\\xff'
        ds.Columns = pixel_array.shape[1]
        ds.Rows = pixel_array.shape[2]
        ds.PixelData = pixel_array[i].tostring()
        dcm = "{0}{1}-{2}.dcm".format(output_path,filename,i)
        print("dcm filename = %s,shape = %s" % (dcm, pixel_array[0].shape))
        ds.save_as(dcm)


def write_count_files(count):
    index = 0
    for fcount, img_file in enumerate((file_list)):
        img_array, b = _read_file_to_numpy(img_file)
        if b:
            index = index + 1
            _write_numpy_2_dicom(img_array, img_file + '.dcm')
        if index > count:
            break

if __name__ == '__main__':
    write_count_files(3)