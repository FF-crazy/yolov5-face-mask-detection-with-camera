import random
import unittest

from balance_data import yolo_to_xyxy, xyxy_to_yolo


class TestBalanceDataConversions(unittest.TestCase):
    def test_round_trip_conversion(self):
        width = 640
        height = 480
        for cls_id in range(3):
            for _ in range(10):
                x1 = random.randint(0, width - 2)
                y1 = random.randint(0, height - 2)
                x2 = random.randint(x1 + 1, width - 1)
                y2 = random.randint(y1 + 1, height - 1)
                original_xyxy = [cls_id, x1, y1, x2, y2]

                yolo_box = xyxy_to_yolo(original_xyxy, width, height)
                converted_xyxy = yolo_to_xyxy(yolo_box, width, height)

                self.assertEqual(converted_xyxy, original_xyxy)


if __name__ == "__main__":
    unittest.main()
