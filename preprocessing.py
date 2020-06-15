import os
import numpy as np
from scipy import ndimage
import pydicom
import trimesh
import cv2
from skimage.measure import regionprops

def create_mask_and_croppped_dicom(patient_id, roi, obj_labels):
    
    stl_dir = '../data_labeled/STLs/'

    obj_names = obj_labels.keys()
    
    patient_id = str(patient_id)
    ArrayDicom = read_dicom(patient_id)
    ArrayDicom_norm = normalize_CT(ArrayDicom)
    mask = np.zeros(ArrayDicom.shape)
    
    for obj_name in obj_labels.keys():
        stl_path = stl_dir+patient_id+'_'+obj_name
        label = obj_labels[obj_name]
        if os.path.exists(stl_path): # check file existence
            tri_mesh = trimesh.load_mesh(stl_path)
            v = tri_mesh.voxelized(pitch=0.4)
            matrix_contour = v.matrix
            matrix_full = ndimage.morphology.binary_fill_holes(matrix_contour)
            o = define_origin(tri_mesh.bounds)
            d1, d2, d3 = matrix_full.shape

            inds = ArrayDicom_norm[o[1]:o[1]+d2,o[0]:o[0]+d1,o[2]:o[2]+d3]*matrix_full.transpose(1,0,2)[:,:,:]!=0
            mask[o[1]:o[1]+d2,o[0]:o[0]+d1,o[2]:o[2]+d3][inds] = label
            
    mask = mask[roi]
    dicom_cropped = ArrayDicom_norm[roi]
    
            
    return mask, dicom_cropped

def pad_zerro(i):
    if len(str(i))==1:
        return '000'+str(i)
    elif len(str(i))==2:
        return '00'+str(i)
    elif len(str(i))==3:
        return '0'+str(i)
    else:
        return str(i)
    
def define_origin(bounds):
    corner_min = bounds[0]
    result = []
    for c in corner_min:
        sl = round((c-0.2)/0.4)
        result.append(int(sl))
    return np.array(result)

def normalize_CT(ArrayDicom):
    array_max = ArrayDicom.max()
    array_min = ArrayDicom.min()
    return (ArrayDicom - array_min)/(array_max-array_min)

def normalize(x):
    xmin=x.min()
    xmax=x.max()
    return 255.*(x-xmin)/(xmax-xmin)

def normalize_3d(img):
    if img.ndim == 2:
        img_norm = (img-img.min())/ max(img.max()-img.min(),1e-8)    
    elif img.ndim == 3:
        channels = [(img[:,:,c]-img[:,:,c].min())/ max(img[:,:,c].max()-img[:,:,c].min(),1e-8) for c in range(img.shape[2])]
        img_norm = np.stack(channels, axis=2)
    return img_norm

    
def save_mask_and_cropped_dicom(path_dic, path_mask, patient_id, roi, obj_labels, slide=100): # slide=100 for all, 70 for balls
    mask, ArrayDic = create_mask_and_croppped_dicom(patient_id, roi, obj_labels)
    
    mask = normalize(mask)
    ArrayDic = normalize(ArrayDic)
    print (mask.shape)
    print (ArrayDic.shape)
    dir_mask = path_mask+patient_id+'/'
    if not os.path.exists(dir_mask):
        os.makedirs(dir_mask)
    for i in range(mask.shape[2]):
        # for cv2.imwrite {0,1} should be mapped to mapped to {0,255}
        cv2.imwrite(dir_mask+pad_zerro(i+1)+".tiff", mask[:,:,i])  # save slide as JPG file
    
    dir_dicom = path_dic+patient_id+'/'
    if not os.path.exists(dir_dicom):
        os.makedirs(dir_dicom)
    for i in range(ArrayDic.shape[2]):
         # for cv2.imwrite {0,1} should be mapped to mapped to {0,255}
        cv2.imwrite(dir_dicom+pad_zerro(i+1)+".tiff", ArrayDic[:,:,i])  # save slide as JPG file
        

def create_mean_mask(obj_labels):
    stl_dir = '../data_labeled/STLs/'
    obj_names = obj_labels.keys()
    Masks = []
    
    for patient_id in range(1, 11):
        patient_id = str(patient_id)
        # read dicom
        ArrayDicom = read_dicom(patient_id)
        # check shape of dicoms 
        print (patient_id, ArrayDicom.shape)
        if ArrayDicom.shape[0] > 686:
            ArrayDicom = ArrayDicom[:686,:686,:686]
        if ArrayDicom.shape[0] < 686: 
            ArrayDicom = np.pad(ArrayDicom,((0,686-ArrayDicom.shape[0]),(0,686-ArrayDicom.shape[0]),(0,686-ArrayDicom.shape[0])),
                               mode = 'constant')

        mask = np.zeros(ArrayDicom.shape)
        ArrayDicom_norm = normalize_CT(ArrayDicom)

        for obj_name in obj_labels.keys():
            stl_path = stl_dir+patient_id+'_'+obj_name
            label = obj_labels[obj_name]
            if os.path.exists(stl_path): # check file existence
                tri_mesh = trimesh.load_mesh(stl_path)
                v = tri_mesh.voxelized(pitch=0.4)
                matrix_contour = v.matrix
                matrix_full = ndimage.morphology.binary_fill_holes(matrix_contour)
                
                o = define_origin(tri_mesh.bounds)
                d1, d2, d3 = matrix_full.shape
    #             !!!! x <-> y, [Y,X,Z]
                inds = ArrayDicom_norm[o[1]:o[1]+d2,o[0]:o[0]+d1,o[2]:o[2]+d3]*matrix_full.transpose(1,0,2)[:,:,:]!=0

                mask[o[1]:o[1]+d2,o[0]:o[0]+d1,o[2]:o[2]+d3][inds] = label
                
                if obj_name in ['RB_001.stl','RM_001.stl','RC_001.stl']:
                    mask_flip = np.flip(mask, axis=1)
                    Masks.append(mask_flip)
                elif obj_name in ['LB_001.stl','LM_001.stl', 'LC_001.stl']:    
                    Masks.append(mask)
                else:
                    mask_flip = np.flip(mask, axis=1)
                    Masks.append(mask_flip)
                    Masks.append(mask)
                
                
    print ('# masks:',len(Masks))                
    return np.array(Masks).mean(axis=0)


def read_dicom(patient_id):
    patient_id  = str(patient_id)
    # read dicom
    PathDicom = '../data_unzip/'+ patient_id +'/data/images/' + patient_id
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(dirName+'/'+filename)
    lstFilesDCM.sort() #!!!!!!
    # Get ref file
    RefDs = pydicom.read_file(lstFilesDCM[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
    # Calculate coordinate axes for this array 
    x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  
            
    return ArrayDicom

def box_coordinate(label_img):
    regions = regionprops(label_img)
    z_min, z_max, y_min, y_max, x_min, x_max = [],[],[],[],[],[]

    areas = [prop.area for prop in regions]
    #sort the array by area
    sorteddata = sorted(zip(areas, regions), key=lambda x: x[0], reverse=True)
    first_largest_region = sorteddata[0][1]
    second_largest_region = sorteddata[1][1]

    for region in [first_largest_region,second_largest_region]:
        z1, y1, x1, z2, y2, x2 = region['bbox']
        z_min.append(z1)
        z_max.append(z2)
        y_min.append(y1)
        y_max.append(y2)
        x_min.append(x1)
        x_max.append(x2)
    z_min = np.array(z_min)
    z_max = np.array(z_max)
    y_min = np.array(y_min)
    y_max = np.array(y_max)
    x_min = np.array(x_min)
    x_max = np.array(x_max) 

    z1, y1, z2, y2 = max(z_min), max(y_min), min(z_max), min(y_max)
    r_x1,  r_x2 = min(x_min), min(x_max)
    l_x1,  l_x2 = max(x_min), max(x_max)
    R_min = z1, y1, r_x1
    R_max = z2, y2, r_x2
    L_min = z1, y1, l_x1
    L_max = z2, y2, l_x2
    box = R_min, R_max, L_min, L_max
    
    return box

def rescale(x, rescale_size):
#   640
    return int((663-23)*x/rescale_size+23)

def box_voi(label_img, rescale_size):
    regions = regionprops(label_img)
    z_min, z_max, y_min, y_max, x_min, x_max = [],[],[],[],[],[]

    areas = [prop.area for prop in regions]
    #sort the array by area
    sorteddata = sorted(zip(areas, regions), key=lambda x: x[0], reverse=True)
    first_largest_region = sorteddata[0][1]
    second_largest_region = sorteddata[1][1]

    for region in [first_largest_region,second_largest_region]:
        z1, y1, x1, z2, y2, x2 = region['bbox']
        z_min.append(z1)
        z_max.append(z2)
        y_min.append(y1)
        y_max.append(y2)
        x_min.append(x1)
        x_max.append(x2)
    z_min = np.array(z_min)
    z_max = np.array(z_max)
    y_min = np.array(y_min)
    y_max = np.array(y_max)
    x_min = np.array(x_min)
    x_max = np.array(x_max) 
    
    z1, z2 = rescale(max(z_min),rescale_size), rescale(min(z_max),rescale_size)
    y1, y2 = rescale(max(y_min),rescale_size), rescale(min(y_max),rescale_size)
    r_x1, r_x2 = rescale(min(x_min),rescale_size), rescale(min(x_max),rescale_size)
    l_x1, l_x2 = rescale(max(x_min),rescale_size), rescale(max(x_max),rescale_size)

    R_voi = (slice(y1, y2, None), slice(r_x1, r_x2,None), slice(z1, z2,None))
    L_voi = (slice(y1, y2, None), slice(l_x1, l_x2,None), slice(z1, z2,None))

    return R_voi, L_voi 