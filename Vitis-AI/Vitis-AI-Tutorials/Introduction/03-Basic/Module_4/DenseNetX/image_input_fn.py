'''
 Copyright 2020 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import os
import cv2


calib_image_list = './build/quantize/images/calib_list.txt'
calib_batch_size = 100

def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  for index in range(0, calib_batch_size):
    curline = line[iter * calib_batch_size + index]
    calib_image_name = curline.strip()

    # open image as BGR
    image = cv2.imread(calib_image_name)

    # change to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # normalize
    image = image/255.0

    images.append(image)
  return {"input_1": images}

