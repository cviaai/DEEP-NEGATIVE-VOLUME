import numpy as np
import os
from preprocessing import pad_zerro, normalize, normalize_3d
import SimpleITK as sitk
import cv2

def create_edges(images, outputs, sigma, upt, lwt, kernel=np.ones((3,3),np.uint8),thresh = 0.1): # 2 ways: Edges of raw images * outputs or Edges of [raw images*outputs]
    im = images[0].permute(1,2,0).data.cpu().numpy()
    out = outputs[0,1].permute(1,2,0).data.cpu().numpy()
    vol_norm_raw = normalize(im)
    vol_norm_raw = np.uint8(vol_norm_raw)
    sitk_img_raw = sitk.GetImageFromArray(vol_norm_raw)
    sitk_img_raw_float = sitk.Cast(sitk_img_raw, sitk.sitkFloat32)
    edges_raw = sitk.CannyEdgeDetection(sitk_img_raw_float, lowerThreshold=lwt, upperThreshold=upt, variance = [sigma,sigma,sigma])
    edges_array_raw = sitk.GetArrayFromImage(edges_raw)
    edges_array_final = ((edges_array_raw*out)>thresh).astype(np.uint8)
    edges_closing = cv2.morphologyEx(edges_array_final, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return edges_closing

def save_edges_output(edges, dir_path, patient_id, object_name):
    edges = normalize(edges)
    edges = edges.astype(np.uint8)
    obj_path = dir_path+str(patient_id)+'/'+object_name+'/'
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)
    for j in range(edges.shape[2]):
    # for cv2.imwrite {0,1} should be mapped to mapped to {0,255}
        cv2.imwrite(obj_path+pad_zerro(j+1)+".tiff", edges[:,:,j])
