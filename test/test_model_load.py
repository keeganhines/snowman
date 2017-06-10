import unittest
from snowman.model.text_model import TextModel

class TestModelLoad(unittest.TestCase):

	def test_load_serialized_model(self):
		model = TextModel()
		model.load()
		test_string = ".switchvpn.net"
		prediction = model.predict(test_string)
		print "Score for test string [" + test_string + "] is: " + str(prediction)
		self.assertTrue( prediction > 0 and prediction < 1)

if __name__ == '__main__':
	unittest.main()