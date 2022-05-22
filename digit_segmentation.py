import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
from itertools import cycle
from shutil import make_archive
import os
from zipfile import ZipFile

header = st.container()
desc = st.container()
image_upload = st.container()
tune = st.expander("Tune the hyper-parameters")
image_pre_process = st.container()
image_box = st.container()
tune_image_size = st.expander("Tune the image size")
image_characters = st.container()
remove_box = st.container()
final_box = st.container()

preprocessed_image = None
img_bounded_final = None





with header:
    st.title("Image Segmentation Project")

with desc:
    st.markdown(
        "This is an Image Segmentation Project. This project aims to provide as a tool to segment characters from a "
        "given image. This project inputs a paper image with handwritten digits written on it. You can preview the "
        "preprocessed image and tune the hyper-parameters if necessary. After that you can preview the cropped "
        "characters' images and remove particular images that you don't need. When you are satisfied with the result, "
        "you can download the zip file containing all the characters' images. Libraries such as 'Streamlit' for "
        "Front-end, 'OpenCV' for Image Processing, 'Numpy' for Mathematical Operations for Array, 'OS', and 'Zipfile' "
        "for creating zip file, are used. The whole project source code can be found 'here'."
    )

with image_upload:
    image = st.file_uploader("Choose An Image File:", type=["jpg", "png", "svg"])





with tune:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        degree = st.slider("Image Rotation Degree", 0, 270, 0, 90)
        st.info("Default Value: 0")
    with col2:
        dilation = st.slider("Kernal Size for Image Dilation", 1, 21, 7, 2)
        st.info("Default Value: 7")
    with col3:
        blur = st.slider("Kernal Size for Image Median Blur", 1, 31, 3, 2)
        st.info("Default Value: 21")
    with col4:
        thresh = st.slider("Threshold Value for Image Thresholding", 100, 250, 200, 5)
        st.info("Default Value: 200")

def rotate(img: np.ndarray, degree: int) -> np.ndarray:
    """
    This function takes a raw image, and degree as inputs and returns a rotated image with input degree.
    Arguments:
      img --  A raw input image
      degree  - Degree to be rotated
    Returns:
      img --  A rotated image with input degree
    """
    if img is not None:
        if degree != 0:
            if degree == 90:
                rotate_flag = cv.ROTATE_90_CLOCKWISE
            elif degree == 180:
                rotate_flag = cv.ROTATE_180
            elif degree == 270:
                rotate_flag = cv.ROTATE_90_COUNTERCLOCKWISE
            img = cv.rotate(img, rotate_flag)
    return img

def bounding(img):
  img_dilation = cv.dilate(img, None, iterations = 2)

  coord_x, coord_xm, coord_y, coord_ym = [], [], [], []
  
  contours, h = cv.findContours(img_dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
  
  for i, ctr in enumerate(contours):

      # Get bounding box
      x, y, w, h = cv.boundingRect(ctr)

      if w > 15 and w < 100 and h > 15 and h < 120:
        coord_x.append(x)
        coord_xm.append(x + w)
        coord_y.append(y)
        coord_ym.append(y + h)
        cv.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
      
  return img, coord_x, coord_xm, coord_y, coord_ym

img = None
with image_box:

    if image:
        img = cv.imread(image.name)
        if img:
            st.write("Uploaded")
        else:
            st.write("Not Uploaded")
        name = image.name
        file_name = name.split('.')[0]

        directory = 'resized_images_'+file_name
  
        # Parent Directory path
        # parent_dir = r"C:\Users\User\Desktop\TPH"
        
        # Path
        # path = os.path.join(parent_dir, directory)
        # if not os.path.exists(path):
        #     os.mkdir(path)

#         result_planes = []

#         rgb_planes = cv.split(img)


#         for plane in rgb_planes:
#             dilated_img = cv.dilate(plane, np.ones((dilation,dilation), np.uint8))
#             bg_img = cv.medianBlur(dilated_img, 7)
#             diff_img = 255 - cv.absdiff(plane, bg_img)
#             norm_img = cv.normalize(diff_img, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX)
#             result_planes.append(norm_img)

#         img_shadow_removed = cv.merge(result_planes)

#         img_gray = cv.cvtColor(cv.UMat(img_shadow_removed), cv.COLOR_BGR2GRAY)


#         ret,img_thres = cv.threshold(img_gray,thresh,255, cv.THRESH_BINARY_INV)

#         img_thres = cv.medianBlur(img_thres,blur)
#         # img_thres4=img_dilation = cv.dilate(img_thres4, np.ones((2,2), np.uint8), iterations=1)

#         img_thres = rotate(img_thres,degree)

#         img_bounded_final, x, xm, y, ym = bounding(img_thres)

#         img_array = cv.UMat.get(img_bounded_final)

#         st.image(img_array)

# resized_imgs = []
# with image_characters:
#     if img_bounded_final!=None:

#         img_thres = cv.UMat.get(img_thres)
#         for k in range(len(x)):
#             i = img_thres[y[k]-2:ym[k]+2, x[k]-2:xm[k]+2]
#             constant= cv.copyMakeBorder(i,10,10,10,10,cv.BORDER_CONSTANT,value=[0,0,0])
#             resize_i = cv.resize(constant,(28,28))
#             resized_imgs.append(resize_i)
#         resized_imgs.reverse()

#         cols = cycle(st.columns(10)) # st.columns here since it is out of beta at the time I'm writing this
#         for idx, filteredImage in enumerate(resized_imgs):
#             next(cols).image(filteredImage,caption = idx)


# nums = [n for n in range(len(resized_imgs))]
# with final_box:
#     with st.form('my_form'):
#         idxs = st.multiselect('Choose the index of character to remove',nums)
#         submitted = st.form_submit_button('Submit')

#     if submitted:
#         idxs.sort()
#         count = 0
#         for i in idxs:
#             resized_imgs.pop(i-count)
#             count+=1

#         cols = cycle(st.columns(10)) # st.columns here since it is out of beta at the time I'm writing this
#         for idx, filteredImage in enumerate(resized_imgs):
#             next(cols).image(filteredImage,caption = idx)
        


#         file_names = ['9','8','7','6','5','4','3','2','1','0']
#         file_names.reverse()
#         count = 0
#         file_no = 0
#         zipObj = ZipFile(directory+'.zip','w')
#         for i,img in enumerate(resized_imgs):
#             if count == 10:
#                 count = 0
#                 file_no += 1
#             data = Image.fromarray(img)
#             data.save(file_name+'_'+file_names[file_no]+str(count)+'.jpg')
#             zipObj.write(file_name+'_'+file_names[file_no]+str(count)+'.jpg',data)
#             count+=1
        
#         zipObj.close()


#         with open((directory+'.zip'), "rb") as fp:
#             btn = st.download_button(
#                 label="Download ZIP",
#                 data=fp,
#                 file_name=directory+'.zip',
#                 mime="application/zip"
#             )
