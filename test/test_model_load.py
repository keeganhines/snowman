import unittest
from snowman.model.text_model import TextModel

class TestModelLoad(unittest.TestCase):

	def test_load_serialized_model(self):
		# model = TextModel()
		# model.load()
		# prediction = model.predict("exampledomain.com")
		# self.assertTrue( prediction > 0 & prediction < 1)

if __name__ == '__main__':
	unittest.main()