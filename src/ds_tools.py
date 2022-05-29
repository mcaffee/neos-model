import os

import settings

import cv2


class CV2Generator:

    def __init__(self, letter):
        self.letter = letter
        self.video_paths = list((settings.USER_SOURCE_DATA_PATH / self.letter).glob('*.mp4'))
        self.counter = 0
        self.output_path = settings.USER_TRAIN_DATA_PATH / letter

    def generate(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cleanup()

        for vpath in self.video_paths:
            print(f'Processing {str(vpath)}')
            vidcap = cv2.VideoCapture(str(vpath))

            while True:
                success, image = vidcap.read()

                if not success:
                    break

                img_path = str(self.output_path / f'frame{self.counter:05d}.jpg')
                cv2.imwrite(img_path, image)
                self.counter += 1

    def cleanup(self):
        for f in self.output_path.glob('*.jpg'):
            os.remove(f)


class Splitter:

    def __init__(self, letter, split=0.15, target='validate'):
        self.letter = letter
        self.split_val = split

        self.source_paths = list(
            sorted(
                (settings.USER_TRAIN_DATA_PATH / self.letter).glob('*.jpg'),
                key=os.path.basename
            )
        )
        if target == 'validate':
            self.output_path = settings.USER_VALIDATE_DATA_PATH / letter
        elif target == 'test':
            self.output_path = settings.USER_TEST_DATA_PATH / letter

    def split(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cleanup()

        split_amount = len(self.source_paths) * self.split_val

        print(f'Rendering to: {str(self.output_path)}')
        print(f'Total files: {len(self.source_paths)}')
        print(f'Files will be split: {int(split_amount)}')

        split_step = int(len(self.source_paths) // (len(self.source_paths) * self.split_val))

        for i in range(0, len(self.source_paths), split_step):
            path = self.source_paths[i]
            new_path = self.output_path / os.path.basename(path)
            os.rename(path, new_path)

    def cleanup(self):
        for f in self.output_path.glob('*.jpg'):
            os.remove(f)


if __name__ == '__main__':
    for letter in settings.CLASSES:
        gen = CV2Generator(letter)
        gen.generate()

        split = Splitter(letter, split=0.15, target='validate')
        split.split()

        split = Splitter(letter, split=0.05, target='test')
        split.split()
