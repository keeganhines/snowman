import unittest
from snowman.model.text_model import TextModel

class TestModelLoad(unittest.TestCase):

	def test_load_serialized_model(self):
		model = TextModel()
		model.train()
		model.save()

if __name__ == '__main__':
	unittest.main()