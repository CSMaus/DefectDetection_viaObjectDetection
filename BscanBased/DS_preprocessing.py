# Here use saved signals in txt files to create a dataset for training and testing
# as Bscan images and labels of defects if they are
import sys

import numpy as np
import os
import cv2
import collections
import json
from tqdm import tqdm

# PARAMETERS

# annotations would be like:
# later I'll add creation multiple sequences for one datafile
# But all annotation will have all prepared datafiles (.opd)
# annotations = {
#     "sequence_001": {
#         "img_001.png": [{"bbox": [beam_DefectStart, beam_DefectEnd, depthStart, depthEnd], "label": "DefectType"}],
#         "img_002.png": [],  # No defects for this b-scan image
#         "img_003.png": [{"bbox": [30, 42, 0.4, 0.5], "label": "Delamination"}]
#     },
#     "sequence_002": {
#         "img_001.png": [],
#         "img_002.png": [{"bbox": [15, 23, 0.15, 0.2], "label": "Delamination"},
#                {"bbox": [30, 42, 0.4, 0.5], "label": "Delamination"}],
#     }
#    .....
# }

# IMAGES DS should match the structure:
# dataset/
# ├── sequence_001/
# │   ├── img_001.png
# │   ├── img_002.png
# │   └── ...
# ├── sequence_002/
# │   ├── img_001.png
# │   ├── img_002.png
# │   └── ...
# └── annotations.json


def resize_image(img, target_size):
    """bilinear interpolation.
    - target_size: tuple (width, height)
    return: numpy_array
    """
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    # return resized_img
    return cv2.rotate(resized_img, cv2.ROTATE_90_CLOCKWISE)

def get_datafile_sequences(ds_path, file_folder):
    file_folder_path = os.path.join(ds_path, file_folder)
    beams = os.listdir(file_folder_path)
    total_num_scans = len(os.listdir(os.path.join(file_folder_path, beams[0])))
    total_num_beams = len(beams)
    one_seq_img_size = (total_num_beams, total_num_scans)
    sequence_name = file_folder  # now will test preparation with one sequence per original datafile

    sequence = {}
    annotation_for_seq = {}

    beams = sorted(beams, key=lambda beam_i: float(beam_i.split('_')[1]))
    beam_start = float(beams[0].split('_')[1])
    beam_end = float(beams[-1].split('_')[1])
    # beam_idx = 0
    for beam in beams:
        beam_idx = float(beam.split('_')[1])
        beam_folder = os.path.join(file_folder_path, beam)
        scan_files = os.listdir(beam_folder)

        # TODO: fix. Here we have incorrect sorting of names
        # TODO: I'll do them so the name of item in seq would be split('_')[0]
        for scan_i in range(len(scan_files)):
            scan_file = scan_files[scan_i]
            scan_key = scan_file.split('_')[0]
            if scan_key not in sequence.keys():
                sequence[scan_key] = []
            # scan_file_path = beam_folder + "//" + scan_file
            scan_file_path = os.path.join(beam_folder, scan_file)
            # let's guess, that all of them are txt (as should be if they were prepared by my SW)
            signal = np.loadtxt(f"{scan_file_path}", dtype=float)
            sequence[scan_key].append(signal)

            # TODO: remake it. There could be multiple defects for one scan with different beams, so we neet to catch it
            if scan_file.split('.')[0].split('_')[1] == 'Health':  # in scan_file
                if f'{scan_key}.png' not in annotation_for_seq.keys():
                    # we meet this scan image first time and there is no defects
                    annotation_for_seq[f'{scan_key}.png'] = []
                    # else: we should not do anything, bcs it might have defect info, and we do not have to rewrite it
            else:
                try:
                    defect_type = scan_file.split('_')[0]
                    defect_start_end = scan_file.split('_')[-1].split('-')
                    defect_start = float(defect_start_end[0])
                    defect_end = float(defect_start_end[1][:-4])

                    if not ("Health" in scan_file):  # i e this scan file name contains defect info
                        if f'{scan_key}.png' not in annotation_for_seq.keys() or len(
                                annotation_for_seq[f'{scan_key}.png']) == 0:
                            # we meet this defect first time, so we have to save it's depth and start beam indexes
                            # but there could be case that we add info for this scan as health, so in this case
                            # we need to create new information about defect
                            annotation_for_seq[f'{scan_key}.png'] = [{
                                "bbox": [beam_idx, beam_idx,
                                         defect_start, defect_end],
                                "label": "Delamination"
                            }]
                        else:
                            condition = bool(annotation_for_seq[f'{scan_key}.png'][-1]["bbox"][2] == defect_start
                                             and annotation_for_seq[f'{scan_key}.png'][-1]["bbox"][3] == defect_end
                                             and annotation_for_seq[f'{scan_key}.png'][-1]["bbox"][1] == beam_idx - 1)
                            # check if the scan_i is in annotation, and it has same depths values, and beam[1] is less than curent in 1
                            if condition:
                                # we met defect before, so we have to increase end beam index
                                annotation_for_seq[f'{scan_key}.png'][-1]["bbox"][1] += 1
                            else:
                                # we met new defect
                                annotation_for_seq[f'{scan_key}.png'].append({
                                    "bbox": [beam_idx, beam_idx,
                                             defect_start, defect_end],
                                    "label": "Delamination"
                                })
                except Exception as ex:
                    print(f"Error: {ex} in {sequence_name}, {beam}")
                    break



            if beam == beams[-1]:
                sequence[scan_key] = np.array(sequence[scan_key])
                # TODO: here need to resize the image and save it as needed.
                # TODO: better to place it into separate function for correct ds folder organizing


    # need to add annotations sequence to bigger dictionary
    annotation_for_seq_sorted = collections.OrderedDict(
        sorted(annotation_for_seq.items(), key=lambda x: int(x[0].split(".")[0])))
    sequence_sorted = collections.OrderedDict(sorted(sequence.items(), key=lambda x: int(x[0].split(".")[0])))
    return sequence_sorted, annotation_for_seq_sorted, (beam_start, beam_end)

def adjust_annotations(annot, beam_lims, size):
    beam_start, beam_end = beam_lims
    beam_len = beam_end - beam_start
    beam_s, depth_s = size
    for scan_i in annot.keys():
        for defect in annot[scan_i]:

            # I added "beam_s -" bcs it was inverted by x axis before, which leads to wrong bbox
            defect["bbox"][0] = int(round(beam_s - beam_s*(defect["bbox"][0] - beam_start)/beam_len))
            defect["bbox"][1] = int(round(beam_s - beam_s*((defect["bbox"][1] - beam_start)/beam_len)))


            defect["bbox"][2] = int(round(defect["bbox"][2]* depth_s))
            defect["bbox"][3] = int(round(defect["bbox"][3] * depth_s))

    return annot

def save_seq_as_images(seq, nn_ds_folder_file):
    for img_name in seq.keys():
        img = seq[img_name]
        img = resize_image(img, (320, 320))
        img = (img * 255).astype(np.uint8)
        img_path = os.path.join(nn_ds_folder_file, f'{img_name}.png')
        cv2.imwrite(img_path, img)

ds_folder = "D:/DataSets/!0_0NaWooDS/2025_DS/"

nn_ds_folder = "dataset/"
file_folders = os.listdir(ds_folder)
annotations = {}
num_saved_files = 0


_file_folder_ = "787-404_07_Ch-0"
def single_file_imgs_processing(ff):
    s, a, bs = get_datafile_sequences(ds_folder, ff)
    nn_ds_ff = os.path.join(nn_ds_folder, ff)
    if not os.path.exists(nn_ds_ff):
        os.makedirs(nn_ds_ff)

    save_seq_as_images(s, nn_ds_ff)
    a = adjust_annotations(a, bs, (320, 320))
    annotations[ff] = a
    with open("annotations_single_file.json", "w") as f:
        json.dump(annotations, f, indent=2)



single_file_imgs_processing(_file_folder_)
sys.exit()
# test!!!
# this file has intersected defects. One is in the bottom (real), one is in the middle (artificially manufactured)
# so the problems appears in this data processing
# file_folder = "787-404_07_Ch-0"
# file_folder_path = os.path.join(ds_folder, file_folder)
# seq, ann = get_datafile_sequences(file_folder_path)
# sys.exit()
# for file_folder in file_folders:

# maybe later better to prepare all this data in c# wpf app
# for file_folder in file_folders:
for file_folder in tqdm(file_folders, desc="Processing", unit="folder", bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} ({percentage:.1f}%)"):
    # file_folder_path = os.path.join(ds_folder, file_folder)
    seq, ann, blims = get_datafile_sequences(ds_folder, file_folder)
    nn_ds_folder_file = os.path.join(nn_ds_folder, file_folder)
    if not os.path.exists(nn_ds_folder_file):
        os.makedirs(nn_ds_folder_file)

    # TODO: uncomment this:
    save_seq_as_images(seq, nn_ds_folder_file)


    # ann should be adjusted also:
    # the beams now as they are, they should be modified to match the correct position of resized image
    # the depths now are %/100 of original signal length - i e image height.
    ann = adjust_annotations(ann, blims, (320, 320))
    annotations[file_folder] = ann
    num_saved_files+=1
    # if num_saved_files == 2:
     #    break


with open("annotations.json", "w") as f:
    json.dump(annotations, f, indent=2)



