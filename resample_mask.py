import argparse
import glob

import SimpleITK as sitk

# parser = argparse.ArgumentParser()
# parser.add_argument('image', metavar='Image', help='Reference image to resample the mask to')
# parser.add_argument('mask', metavar='Mask', help='Input mask to resample')
# parser.add_argument('resMask', metavar='Out', help='Filename to store resampled mask')

def main():
    image_list = glob.glob('/data1/WSI/Pathology_Radiology/Dataset/SYSU/rad/*.dcm')
    mask_list = glob.glob('/data1/WSI/Pathology_Radiology/Dataset/SYSU/rad/*.nii')
    image_list, mask_list = sorted(image_list), sorted(mask_list)
    for (i,m) in zip(image_list, mask_list):
        image = sitk.ReadImage(i)
        mask = sitk.ReadImage(m)
        rif = sitk.ResampleImageFilter()
        rif.SetReferenceImage(image)
        rif.SetOutputPixelType(mask.GetPixelID())
        rif.SetInterpolator(sitk.sitkNearestNeighbor)
        resMask = rif.Execute(mask)
        name = m.split('/')[-1]
        sitk.WriteImage(resMask, '/home/yhchen/Documents/trial/'+name, True)

if __name__ == '__main__':
    main()