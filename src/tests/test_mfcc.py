import unittest
from src.utils import mfcc_extractor


class TestMfcc(unittest.TestCase):
    def test_signal_variables(self):
        """
        Expected output is a tuple: (samples_per_segment, num_mfcc_vectors_per_segment)
        :return:
        """
        self.assertEqual(mfcc_extractor.signal_variables(samples_per_track=1000, num_segments=10, hop_length=1000),
                         (100, 1))
        self.assertEqual(mfcc_extractor.signal_variables(samples_per_track=1000, num_segments=1000, hop_length=10000),
                         (1, 1))
        self.assertEqual(mfcc_extractor.signal_variables(samples_per_track=1000, num_segments=10000, hop_length=10000),
                         (0, 0))


if __name__ == "__main__":
    unittest.main()
