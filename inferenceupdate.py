global parentdir
##Convert files pdf and png images of a selected directory in jpg format"""
import converter
#converter.convertToJpg('/home/ec2-user/SageMaker/teracloud-ai/DocumentAI/Model0621/TestImage')
## importing os to access system level modules
import os 
## Change current directory to below defined model path to access all dependencies and files
Modelpath = '/Volumes/Extreme SSD/teracloud23june/teracloud-ai/DocumentAI/Model0621'
os.chdir(Modelpath)
path = Modelpath
parentdir = os.path.abspath(os.path.join(path, os.pardir))
## import easyocr to perform ocr operations on an image
import easyocr
## below line will download the recognition model and map ocr reader to an object reader 
## setting gpu false so the inference will run only on cpu
reader = easyocr.Reader(['en'], gpu = False) ## gpu false
## Imaging Python Imaging Library to read and perform operations on images
import PIL
## Dataframe package to create dataframe of extracted data
import pandas as pd
## importing json to write the final output in json format
import json
## importing glob to read all files from a directory
import glob
## csv to write or convert fileinto csv format
import csv
## argument parse to accept the values of argument from CLI
import argparse
## Below are all model depencies
## training all architectures on torch platform
import torch
## tqdm to track the progress of live operation
from tqdm import tqdm
## importing the package for path maniplation
from pathlib import Path
## Package to load data into pytorch model required format
from torch.utils.data.dataloader import DataLoader
## Package for CRF Layer 
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from parse_config import ConfigParser
# package to save the model 
import model.pick as pick_arch_module
## package for dataset operations
from data_utils.pick_dataset import PICKDataset
## package for batch operations in current SDK
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str
## package for converting the pdfs to images
import converter
## package for simple copy operation
import shutil
## used to flatten the nested list items
from itertools import chain
device = torch.device(f'cuda:{args.gpu}' if -1 != -1 else 'cpu') ### setting value of gpu to -1 to run inference on cpu
savedCheckpiont = 'saved/models/PICK_Default/test_999/model_best.pth'
checkpoint = torch.load(savedCheckpiont, map_location=device)
config = checkpoint['config']
state_dict = checkpoint['state_dict']
monitor_best = checkpoint['monitor_best']
print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(savedCheckpiont, monitor_best))
# prepare model for testing
pick_model = config.init_obj('model_arch', pick_arch_module)
pick_model = pick_model.to(device)
pick_model.load_state_dict(state_dict)
pick_model.eval()
## pick ocr transcript file and image in below folders
out_img_path = "test_img/"
out_box_path = "test_boxes_and_transcripts/"

def generateTranscript():
    ### convert image into transcript file 
    """Select jpg or jpeg files from a given directory and convert those into transcript files"""
    types = ('*.jpg', '*.jpeg')
    filenames = []
    for files in types:
        filenames.extend(glob.glob(os.path.join(str(parentdir)+"/TestImage/",files)))
    filenames.sort()
    def flatten(listOfLists):
        "Flatten one level of nesting"
        return chain.from_iterable(listOfLists)
    for fname in filenames:
        filen = fname.split(".")[0]
        bounds = reader.readtext(fname)
        df = pd.DataFrame()
        CoordinatesWithValue=[]
        for i in bounds:
            converted_list = [str(element) for element in list(flatten(i[0]))]
            CoordinatesWithValue=converted_list
            CoordinatesWithValue.append(i[1])
            temp_df = pd.DataFrame([CoordinatesWithValue])
            df = df.append(temp_df)
        df.insert(0,'imageNum','') ## inserting new column and setting value 1 to match the format required by the model
        df['imageNum'] = 1
        df.to_csv(str(filen)+".tsv",sep = ',',index=False,header=False, quotechar='',escapechar='\\',quoting=csv.QUOTE_NONE, )
    ### copy file from source folder to temporary folder test_img and text_bAt folder ###
    for f in filenames:
        shutil.copy(f, str(parentdir)+'/PICK-pytorch/test_img/')
    filetsv = glob.glob(str(parentdir)+"/TestImage/*.tsv")
    for f in filetsv:
        shutil.copy(f, str(parentdir)+'/PICK-pytorch/test_boxes_and_transcripts/')

def runInference():
    ### inference code ###
    batch_size_val=1
    test_dataset = PICKDataset(boxes_and_transcripts_folder=out_box_path,
                                images_folder=out_img_path,
                                resized_image_size=(480, 960),
                                ignore_error=False,
                                training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False,
                                    num_workers=0, collate_fn=BatchCollateFn(training=False)) ## have changed the number of workers to zero
    # setup output path
    output_folder = 'output'
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for step_idx, input_data_item in enumerate(test_data_loader):
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)
            # For easier debug.
            image_names = input_data_item["filenames"]
            output = pick_model(**input_data_item)
            logits = output['logits']  # (B, N*T, out_dim)
            new_mask = output['new_mask']
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            mask = input_data_item['mask']
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)
            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)
            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                spans = bio_tags_to_spans(decoded_tags, [])
                spans = sorted(spans, key=lambda x: x[1][0])
                entities = []  # exists one to many case
                dickey = []
                dicvalue=[]
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)
                    tt = text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1])
                    dickey.append(entity_name)
                    dicvalue.append(tt)
                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                result_filejson = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.json')
                ### printing model output and results in json 
                with open(result_filejson, 'w', encoding='utf-8') as f:
                    res = {dickey[i]: dicvalue[i] for i in range(len(dickey))}
                    json.dump(res, f, ensure_ascii=False, indent=4)
                    for key, value in res.items():
                        print(str(key)+" :", value)
        try:
            dir = out_img_path
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f)) ## removing temporary image files 
        except:
            pass
        try:
            dir = out_box_path
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f)) ## removing temporary transcripts file
        except:
            pass
