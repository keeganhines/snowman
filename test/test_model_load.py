import unittest
from snowman.model.text_model import TextModel

class TestModelLoad(unittest.TestCase):

	test_strings = [".rltnsk.biz",
				".android.apps.plus",
				".edxutilities.click",
				".ganadineroconencuestas.com", 
				".books.google.com", 
				".survey.godaddy.com",
				".ru.archive.ubuntu.com",
				".ubuntu.com"]

	def test_load_serialized_model(self):
		model = TextModel()
		model.load()

		for test_string in self.test_strings:
			prediction = model.predict(test_string)
			print "Score for test string [" + test_string + "] is: " + str(prediction)
			self.assertTrue( prediction > 0 and prediction < 1)

if __name__ == '__main__':
	unittest.main()