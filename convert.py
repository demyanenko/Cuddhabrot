#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def convert_c(name):
    pixel_type = np.uint64
    side = int(np.sqrt(os.stat(name).st_size / np.dtype(pixel_type).itemsize))
    cbresult = np.fromfile(name, dtype=pixel_type).reshape((side, side))
    
    # print(np.max(cbresult))
    visual_max = np.percentile(cbresult.flatten(), 99.9)
    # print(visual_max)
    
    part_side = 8000
    
    def save(path, pic):
        plt.imsave(path, pic, cmap='binary_r', vmin=1, vmax=visual_max)
    if side > part_side:
        for i in range(0, side, part_side):
            for j in range(0, side, part_side):
                save("%s_%03i_%03i.tiff" % (name[:-4], i/part_side, j/part_side), cbresult[i:i+part_side, j:j+part_side])
    else:
        save("%s.tiff" % name[:-4], cbresult)

    
if len(sys.argv) == 1:
    convert_c('pic.bin')
else:
    map(convert_c, sys.argv[1:])
