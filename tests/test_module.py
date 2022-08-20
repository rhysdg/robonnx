import os
import unittest
from robonnx.roberta_onnx import OnnxSession 


class RobTestExtra(unittest.TestCase):
    def assertIsFile(self, path):
        if not os.path.isfile(path):
            raise AssertionError(f"{path} does not exist")

class RobTestMain(RobTestExtra):

  #super handy for recurring import and instantiation!
  def setUp(self):
    self.onx = OnnxSession('model.onnx', 'emotion')

  def test_labels(self):
        self.assertEqual(self.onx.labels, ['anger', 'joy', 'optimism', 'sadness'])

  def test_modeldl(self):
        self.assertIsFile(self.onx.model)
  
  
if __name__ == '__main__':
    unittest.main()
