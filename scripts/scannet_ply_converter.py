import os
import argparse
from plyfile import PlyData, PlyElement
import numpy as np


def filter_ply_by_valid_labels(input_file, valid_class_ids):
    input_dir = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    ply_data = PlyData.read(input_file)
    vertex_data = np.array(ply_data['vertex'].data)
    labels = vertex_data['label']
    labels_name = []
    for label_value in valid_class_ids:
        filtered_vertices = vertex_data[labels == label_value]
        if len(filtered_vertices) > 0:
            index = np.where(valid_class_ids == label_value)[0][0]
            output_label = CLASS_LABELS[index]
            labels_name.append(output_label)
            new_vertex_element = PlyElement.describe(filtered_vertices, 'vertex')
            new_ply_data = PlyData([new_vertex_element], text=True)
            base_name = input_filename.split('.ply')[0]
            output_file = os.path.join(input_dir, f'{base_name}_{output_label}.ply')
            new_ply_data.write(output_file)
            print(f"Filtered PLY file for {output_label} saved as {output_file}")
    output_txt_file = os.path.join(input_dir, 'classes.txt')
    with open(output_txt_file, 'w') as f:
        f.write(' '.join(labels_name))
    print(f"Label names written to {output_txt_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter PLY file by valid labels.")
    parser.add_argument('--input_ply', type=str, help="Input PLY file path")
    args = parser.parse_args()

    VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36])
    CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                    'bathtub']
                    
    filter_ply_by_valid_labels(args.input_ply, VALID_CLASS_IDS)