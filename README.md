# Extractor-Suite
Speed up your business processes by automating information extraction with AI. Manage your forms/scanned images, scattered or unmanageable data with our Extractor-Suite by transforming it from messy or unstructured data into digital structured format in no time.

## ![Scanned Documents](https://github.com/Neerajcerebrum/Extractor-Suite/blob/develop/images/detect.png) 

Extractor SDK works on form identification, maintain semantic information, noise reduction and much more to eliminate manual and costly data entry process.


Pack of pre-built deep learning models which extract the crucial and relevant information from any type of data on real time basis. The cutting-edge engines uses both text segments and image BB features for learning and maintain semantic relations between embeddings by using advance graph learning approaches.


## Model Implementation:
1)	Transformers and CNN layer:- With the help of images and OCR engine, Transformer and CNN layers extract features from both images and text segments and combine their embeddings which act as a node and passes to GCN layer to get richer graph embeddings for the downstream task.  
2)	GCNs: Advance graph learning module layer which automatically learn the relations between nodes and uses both text and image features of documents including text, position, layout and image to get an intense semantic representation which is important for extracting the key information without ambiguity. 
3)	BiLSTM and CRF layer: This layer adds sequence tagging on the union non-local sentence at character-level using BiLSTM and CRF, respectively. In this way, the model transforms key information extraction tasks into a sequence tagging problem by considering the layout information and the global information of the document.
4)	The first two modules act as encoder which provides enriched embeddings of both text and image features to the last module which act as a decoder which extract the information by converting the task into sequence tagging problem. 

## Overview of ProcessFlow
## ![Process Flow](https://github.com/Neerajcerebrum/Extractor-Suite/blob/develop/images/Flow.png) 

## Model Training with 125 images and 100 epochs 
## ![Training Process](https://github.com/Neerajcerebrum/Extractor-Suite/blob/develop/images/ModelTraining.png) 

## Model Inference
## ![Training Process](https://github.com/Neerajcerebrum/Extractor-Suite/blob/develop/images/ModelOutput.png)
Though the dataset was quite small, still model able to learn and predict the entities with decent accuracy scores.


