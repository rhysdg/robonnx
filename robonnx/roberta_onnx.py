import os
import csv
import time
import gdown
import urllib.request

import numpy as np
import onnxruntime

from scipy.special import softmax
from transformers import AutoTokenizer
from robonnx.tools import Timer

class OnnxSession(Timer):
  def __init__(self, model, task, verbose=False):
    super(OnnxSession, self).__init__()

    self.model = model
    self.verbose = verbose
    self.task = task
    self.transformer = f"cardiffnlp/twitter-roberta-base-{task}"

    self.labels = self.__setup()
    self.session = self.__onnx_setup()


  @staticmethod
  def __preprocess(text):
      new_text = []
      for t in text.split(" "):
          t = '@user' if t.startswith('@') and len(t) > 1 else t
          t = 'http' if t.startswith('http') else t
          new_text.append(t)
      return " ".join(new_text)
    

  def printm(self, msg):
    if self.verbose:
      print(msg)


  def __onnx_setup(self):
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

    self.printm(f"onnxruntime device: {onnxruntime.get_device()}\n") # output: GPU

    return session


  def __setup(self):
    # download label mapping
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{self.task}/mapping.txt"

    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    if not os.path.isfile(self.model):
      self.printm('\nNo model available. Downloading to model.onnx')
      gdown.download(id='1m6JQyVKh3QobLeBpUI5BXWbMrHS1l3cN')

    self.tokenizer = AutoTokenizer.from_pretrained(self.transformer)
    
    self.printm(f'Using labels: {labels}, for task: {self.task} detection\n')
    
    return labels

  
  def infer(self, text, warmup=200, bench=False):

    #preprocessing
    text = self.__preprocess(text)
    encoded_input = self.tokenizer(text, return_tensors='np')

    if bench:
      for i in range(warmup):
        _ = self.session.run(output_names=["logits"], input_feed=dict(encoded_input))
    
    self.start()
    outputs = self.session.run(output_names=["logits"], input_feed=dict(encoded_input))
    self.end()
    
    scores = softmax(outputs)[0][0]
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    res = {self.labels[ranking[i]]:np.round(float(scores[ranking[i]]), 4) for i in range(scores.shape[0])}
  
    return res


if __name__ == '__main__':
  
  onx = OnnxSession('model.onnx', 'emotion', verbose=False)
  fake_tweet = '@rhysdg apparently you can install all your work in one line! The robots will be taking our jobs soon. Sign the petiton at http://githubissentient.com'
  
  res = onx.infer(fake_tweet)
  print(res)
