import numpy as np
from radiomics import featureextractor
import six
import glob

from torch_geometric.io.tu import names

# imagePath, maskPath = radiomics.getTestCase('brain1')
imagePath = '/data1/WSI/Pathology_Radiology/Dataset/SYSU/rad/*.dcm'
maskPath = '/data1/WSI/Pathology_Radiology/Dataset/SYSU/rad/*.nii'
# if imagePath is None or maskPath is None:  # Something went wrong, in this case PyRadiomics will also log an error
#     raise Exception('Error getting testcase!')  # Raise exception to prevent cells below from running in case of "run all"
image_list = sorted(glob.glob(imagePath))
mask_list = sorted(glob.glob(maskPath))
names_exclude = []

extractor = featureextractor.RadiomicsFeatureExtractor(sigma=[1, 2, 3], correctMask=True, force2D=True)
extractor.enableImageTypeByName('Gradient')
extractor.enableImageTypeByName('Wavelet')
extractor.enableImageTypeByName('Square')
extractor.enableImageTypeByName('LBP2D')

for (i,m) in zip(image_list, mask_list):
    name = i.split('/')[-1].split('.')[0]
    try:
        result = extractor.execute(i,m)
    except:
        names_exclude.append(name)
        continue
    feat_original = []
    feat_gd = []
    feat_Wavelet = []
    feat_Square = []
    feat_lbp = []
    for key, value in six.iteritems(result):
        if key.startswith('original'):
            feat_original.append(value)
        elif key.startswith('gradient'):
            feat_gd.append(value)
        elif key.startswith('wavelet'):
            feat_Wavelet.append(value)
        elif key.startswith('square'):
            feat_Square.append(value)
        elif key.startswith('lbp'):
            feat_lbp.append(value)
    feat = feat_original + feat_gd + feat_Wavelet + feat_Square + feat_lbp
    feat = np.array(feat)
    assert feat.shape[0] == 758
    np.save( f'/data1/yhchen/sysu_pyradiomics/{name}.npy',feat)

print(names_exclude)
# exlude_list = ['10290723', '10377959', '10396806', '10422107']
# print('Extraction parameters:\n\t', extractor.settings)
# print('Enabled filters:\n\t', extractor.enabledImagetypes)
# print('Enabled features:\n\t', extractor.enabledFeatures)



# print('Result type:', type(result))  # result is returned in a Python ordered dictionary)
# print('')
# print('Calculated features')
# for key, value in six.iteritems(result):
#     print('\t', key, ':', value)