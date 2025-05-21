# here take json files and same its as images and annotation for them
# todo: solve the problem with the fillu zeroes images - how does they appear
# todo: it might be error of scanning - need to note it in NN training
import os
import json
import numpy as np
import cv2
import collections
from tqdm import tqdm

number_false_imgs = 0

# ds_folder = "D:/DataSets/!0_0NaWooDS/2025_DS/WOT_JSON/"
ds_folder = "WOT-20250521/"
nn_ds_folder = "dataset/WOT-20250501/"
if not os.path.exists(nn_ds_folder):
    os.makedirs(nn_ds_folder)
annotation_file_name = "annotations-WOT-20250501.json"

def resize_image(img, target_size):
    """bilinear interpolation.
    - target_size: tuple (width, height)
    return: numpy_array
    """
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    # return resized_img
    return cv2.rotate(resized_img, cv2.ROTATE_90_CLOCKWISE)


def get_datafile_sequences(ds_path, json_filename):
    file_path = os.path.join(ds_path, json_filename)
    with open(file_path, "r") as f:
        data = json.load(f)

    sequence_name = os.path.splitext(json_filename)[0]
    beams = sorted(data.keys(), key=lambda beam_i: float(beam_i.split('_')[1]))
    beam_start = float(beams[0].split('_')[1])
    beam_end = float(beams[-1].split('_')[1])

    sequence = {}
    annotation_for_seq = {}

    for beam in beams:
        beam_idx = float(beam.split('_')[1])
        scan_dict = data[beam]
        scan_files = list(scan_dict.keys())

        for scan_file in scan_files:
            scan_key = scan_file.split('_')[0]
            if scan_key not in sequence:
                sequence[scan_key] = []

            signal = scan_dict[scan_file]
            sequence[scan_key].append(signal)

            if scan_file.split('_')[1] == 'Health':
                if f'{scan_key}.png' not in annotation_for_seq.keys():
                    annotation_for_seq[f'{scan_key}.png'] = []
            else:
                try:
                    defect_start_end = scan_file.split('_')[-1].split('-')
                    defect_start = float(defect_start_end[0])
                    defect_end = float(defect_start_end[1])

                    if f'{scan_key}.png' not in annotation_for_seq.keys() or len(
                            annotation_for_seq[f'{scan_key}.png']) == 0:
                        lbl = scan_file.split('_')[1]
                        annotation_for_seq[f'{scan_key}.png'] = [{
                            "bbox": [beam_idx, beam_idx, defect_start, defect_end],
                            "label": lbl
                        }]
                    else:
                        condition = bool(annotation_for_seq[f'{scan_key}.png'][-1]["bbox"][2] == defect_start
                                         and annotation_for_seq[f'{scan_key}.png'][-1]["bbox"][3] == defect_end
                                         and annotation_for_seq[f'{scan_key}.png'][-1]["bbox"][1] == beam_idx - 1)
                        # check if the scan_i is in annotation, and it has same depths values, and beam[1] is less than curent in 1
                        if condition:
                            annotation_for_seq[f'{scan_key}.png'][-1]["bbox"][1] += 1
                        else:
                            # we met new defect
                            annotation_for_seq[f'{scan_key}.png'].append({
                                "bbox": [beam_idx, beam_idx,
                                         defect_start, defect_end],
                                "label": "Delamination"
                            })

                        '''
                        last_def = annotation_for_seq[f'{scan_key}.png'][-1]
                        if last_def["bbox"][2] == defect_start and last_def["bbox"][3] == defect_end and last_def["bbox"][1] == beam_idx - 1:
                            last_def["bbox"][1] += 1
                        else:
                            lbl = scan_file.split('_')[1]
                            annotation_for_seq[f'{scan_key}.png'].append({
                                "bbox": [beam_idx, beam_idx, defect_start, defect_end],
                                "label": lbl
                            })
                        '''
                except Exception as ex:
                    print(f"Error: {ex} in {sequence_name}, {beam}, {scan_file}")
                    break

        if beam == beams[-1]:
            for scan_key in sequence:
                sequence[scan_key] = np.array(sequence[scan_key])

    annotation_for_seq_sorted = collections.OrderedDict(
        sorted(annotation_for_seq.items(), key=lambda x: int(x[0].split(".")[0])))
    sequence_sorted = collections.OrderedDict(
        sorted(sequence.items(), key=lambda x: int(x[0].split(".")[0])))
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
    global number_false_imgs
    for img_name in seq.keys():
        img = seq[img_name]
        try:
            img = resize_image(img, (320, 320))
        except Exception as ex:
            # print(f"Error: {ex} in {nn_ds_folder_file}, {img_name}")
            number_false_imgs += 1
            continue
        img = (img * 255).astype(np.uint8)
        img_path = os.path.join(nn_ds_folder_file, f'{img_name}.png')
        cv2.imwrite(img_path, img)



annotations = {}
json_files = [f for f in os.listdir(ds_folder) if f.endswith('.json')]

for json_file in tqdm(json_files, desc="Processing JSON", unit="file"):
    json_path = os.path.join(ds_folder, json_file)
    base_name = os.path.splitext(json_file)[0]

    seq, ann, blims = get_datafile_sequences(ds_folder, json_file)

    nn_ds_folder_file = os.path.join(nn_ds_folder, base_name)
    if not os.path.exists(nn_ds_folder_file):
        os.makedirs(nn_ds_folder_file)
    save_seq_as_images(seq, nn_ds_folder_file)

    ann = adjust_annotations(ann, blims, (320, 320))
    annotations[base_name] = ann


with open(annotation_file_name, "w") as f:
    json.dump(annotations, f, indent=2)

print("Number of images with values only zeroes is: ", number_false_imgs)
