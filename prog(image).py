import threading
import time
import os
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from PIL import Image
from collections import deque


class FilterType(Enum):
    INVERT = "invert"
    GRAYSCALE = "grayscale"
    SEPIA = "sepia"
    BLUR = "blur"
    CONTRAST = "contrast"


@dataclass
class ImageTask:
    task_id: int
    input_path: str
    output_path: str
    filter_type: FilterType

@dataclass
class ImageResult:
    task_id: int
    output_path: str
    success: bool
    error_message: str = ""
    processing_time: float = 0.0


class BlockingQueue:
    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self.queue = deque()
        self.mutex = threading.Lock()
        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)

    def put(self, item):
        with self.not_full:
            while 0 < self.maxsize <= len(self.queue):
                self.not_full.wait()

            self.queue.append(item)
            self.not_empty.notify()

    def get(self, timeout=None):
        with self.not_empty:
            if timeout == 0 and len(self.queue) == 0:
                return None

            if timeout is not None:
                end_time = time.time() + timeout
                while len(self.queue) == 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return None
                    self.not_empty.wait(remaining)
            else:
                while len(self.queue) == 0:
                    self.not_empty.wait()

            item = self.queue.popleft()

            if self.maxsize > 0 and len(self.queue) == self.maxsize - 1:
                with self.not_full:
                    self.not_full.notify()

            return item


class ImageProcessor:

    @staticmethod
    def invert_pixel(r: int, g: int, b: int) -> Tuple[int, int, int]:
        return 255 - r, 255 - g, 255 - b

    @staticmethod
    def grayscale_pixel(r: int, g: int, b: int) -> Tuple[int, int, int]:
        gray = (r + g + b) // 3
        return gray, gray, gray

    @staticmethod
    def sepia_pixel(r: int, g: int, b: int) -> Tuple[int, int, int]:
        tr = int(0.393 * r + 0.769 * g + 0.189 * b)
        tg = int(0.349 * r + 0.686 * g + 0.168 * b)
        tb = int(0.272 * r + 0.534 * g + 0.131 * b)

        return min(255, tr), min(255, tg), min(255, tb)

    @staticmethod
    def contrast_pixel(r: int, g: int, b: int, factor: float = 1.5) -> Tuple[int, int, int]:
        def contrast(val):
            new_val = int((val - 128) * factor + 128)
            return max(0, min(255, new_val))

        return contrast(r), contrast(g), contrast(b)

    @classmethod
    def apply_filter(cls, pixels, width, height, filter_type: FilterType, start_x=0, start_y=0, end_x=None, end_y=None):

        if end_x is None:
            end_x = width
        if end_y is None:
            end_y = height

        new_pixels = []

        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                idx = y * width + x
                if idx >= len(pixels):
                    continue

                r, g, b = pixels[idx][:3]

                if filter_type == FilterType.INVERT:
                    new_rgb = cls.invert_pixel(r, g, b)
                elif filter_type == FilterType.GRAYSCALE:
                    new_rgb = cls.grayscale_pixel(r, g, b)
                elif filter_type == FilterType.SEPIA:
                    new_rgb = cls.sepia_pixel(r, g, b)
                elif filter_type == FilterType.CONTRAST:
                    new_rgb = cls.contrast_pixel(r, g, b)

                new_pixels.append(new_rgb)

        return new_pixels


class Producer(threading.Thread):
    def __init__(self, task_queue: BlockingQueue, image_files: List[str], output_dir: str, filter_type: FilterType, poison_pills_count: int):
        super().__init__()
        self.task_queue = task_queue
        self.image_files = image_files
        self.output_dir = output_dir
        self.filter_type = filter_type
        self.poison_pills_count = poison_pills_count
        self.daemon = False
        self.processed_count = 0

        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        try:
            for idx, input_path in enumerate(self.image_files):
                base_name = os.path.basename(input_path)
                name_without_ext = os.path.splitext(base_name)[0]
                output_filename = f"{name_without_ext}_{self.filter_type.value}.png"
                output_path = os.path.join(self.output_dir, output_filename)

                task = ImageTask(
                    task_id=idx,
                    input_path=input_path,
                    output_path=output_path,
                    filter_type=self.filter_type
                )

                self.task_queue.put(task)
                self.processed_count += 1

        except Exception as e:
            print(f"Error in Producer: {e}")

        finally:
            for i in range(self.poison_pills_count):
                self.task_queue.put(None)


class Consumer(threading.Thread):
    def __init__(self, consumer_id: int, task_queue: BlockingQueue, results_queue: BlockingQueue):
        super().__init__()
        self.consumer_id = consumer_id
        self.task_queue = task_queue
        self.results_queue = results_queue
        self.processed_count = 0
        self.running = True

    def run(self):
        while True:
            try:
                task = self.task_queue.get()

                if task is None:
                    break

                start_time = time.time()
                success, error_msg = self.process_image(task)
                proc_time = time.time() - start_time

                result = ImageResult(
                    task_id=task.task_id,
                    output_path=task.output_path,
                    success=success,
                    error_message=error_msg,
                    processing_time=proc_time
                )

                self.results_queue.put(result)
                self.processed_count += 1

            except Exception as e:
                print(f"Error in Consumer-{self.consumer_id}: {e}")
                continue

    def process_image(self, task: ImageTask):
        try:
            with Image.open(task.input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                pixels = list(img.getdata())
                width, height = img.size

                if task.filter_type == FilterType.BLUR:
                    new_pixels = self.apply_gaussian_blur(pixels, width, height, radius=4.0)
                else:
                    new_pixels = ImageProcessor.apply_filter(
                        pixels, width, height, task.filter_type
                    )

                new_img = Image.new('RGB', (width, height))
                new_img.putdata(new_pixels)

                new_img.save(task.output_path, 'PNG')

                return True, ""

        except Exception as e:
            return False, str(e)

    def apply_gaussian_blur(self, pixels, width, height, radius=4.0):
        kernel_size = int(2 * radius + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = self.generate_gaussian_kernel(radius, kernel_size)
        kernel_sum = sum(kernel)

        kernel = [w / kernel_sum for w in kernel]

        temp_pixels = self.convolve_horizontal(pixels, width, height, kernel)
        final_pixels = self.convolve_vertical(temp_pixels, width, height, kernel)

        result = []
        for i in range(len(final_pixels)):
            r = max(0, min(255, int(final_pixels[i][0])))
            g = max(0, min(255, int(final_pixels[i][1])))
            b = max(0, min(255, int(final_pixels[i][2])))
            result.append((r, g, b))

        return result

    def generate_gaussian_kernel(self, sigma: float, size: int) -> List[float]:
        center = size // 2
        kernel = []

        for i in range(size):
            x = i - center
            g = math.exp(-(x * x) / (2 * sigma * sigma)) / (sigma * math.sqrt(2 * math.pi))
            kernel.append(g)

        return kernel

    def convolve_horizontal(self, pixels, width, height, kernel):
        kernel_size = len(kernel)
        half_kernel = kernel_size // 2
        result = []

        for y in range(height):
            for x in range(width):
                r_sum = 0.0
                g_sum = 0.0
                b_sum = 0.0

                for k in range(kernel_size):
                    nx = x + (k - half_kernel)

                    if nx < 0:
                        nx = -nx - 1
                    elif nx >= width:
                        nx = 2 * width - nx - 1

                    idx = y * width + nx
                    r, g, b = pixels[idx][:3]

                    weight = kernel[k]
                    r_sum += r * weight
                    g_sum += g * weight
                    b_sum += b * weight

                result.append((r_sum, g_sum, b_sum))

        return result

    def convolve_vertical(self, temp_pixels, width, height, kernel):
        kernel_size = len(kernel)
        half_kernel = kernel_size // 2
        result = []

        for y in range(height):
            for x in range(width):
                r_sum = 0.0
                g_sum = 0.0
                b_sum = 0.0

                for k in range(kernel_size):
                    ny = y + (k - half_kernel)

                    if ny < 0:
                        ny = -ny - 1
                    elif ny >= height:
                        ny = 2 * height - ny - 1

                    idx = ny * width + x
                    r, g, b = temp_pixels[idx]

                    weight = kernel[k]
                    r_sum += r * weight
                    g_sum += g * weight
                    b_sum += b * weight

                result.append((r_sum, g_sum, b_sum))

        return result


class ResultsCollector(threading.Thread):
    def __init__(self, results_queue: BlockingQueue, total_tasks: int):
        super().__init__()
        self.results_queue = results_queue
        self.total_tasks = total_tasks
        self.results = []
        self.daemon = False

    def run(self):
        completed = 0
        max_wait = 10
        start_time = time.time()
        while completed < self.total_tasks:
            if time.time() - start_time > max_wait:
                break

            try:
                result = self.results_queue.get()
                self.results.append(result)
                completed += 1
                start_time = time.time()

            except Exception:
                continue


def find_images(directory: str, extensions=None) -> List[str]:
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    images = []
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in extensions):
            images.append(os.path.join(directory, file))
    return images


def print_results_summary(results: List[ImageResult]):
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    print(f"\nProcessing completed:")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")


def main():
    INPUT_DIR = "./input_images"
    OUTPUT_DIR = "./output_images"
    NUM_CONSUMERS = 4

    print("\nAvailable filters:")
    for i, filter_type in enumerate(FilterType):
        print(f"  {i + 1}. {filter_type.value}")

    choice = input("\nSelect the filter to process: ").strip()

    filter_map = {
        '1': FilterType.INVERT,
        '2': FilterType.GRAYSCALE,
        '3': FilterType.SEPIA,
        '4': FilterType.BLUR,
        '5': FilterType.CONTRAST
    }

    selected_filter = filter_map.get(choice, FilterType)
    print(f"The filter is selected: {selected_filter.value}")

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        create_test_image(os.path.join(INPUT_DIR, "test_image.png"))

    images = find_images(INPUT_DIR)

    if not images:
        create_test_image(os.path.join(INPUT_DIR, "test_image.png"))
        images = find_images(INPUT_DIR)

    task_queue = BlockingQueue(maxsize=10)
    results_queue = BlockingQueue()

    producer = Producer(task_queue, images, OUTPUT_DIR, selected_filter, NUM_CONSUMERS)

    consumers = []
    for i in range(NUM_CONSUMERS):
        consumer = Consumer(i, task_queue, results_queue)
        consumers.append(consumer)

    collector = ResultsCollector(results_queue, len(images))

    start_time = time.time()

    collector.start()
    for consumer in consumers:
        consumer.start()
    producer.start()

    producer.join()

    for consumer in consumers:
        consumer.join()

    collector.join()

    total_time = time.time() - start_time

    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print_results_summary(collector.results)
    print(f"Results saved in: {OUTPUT_DIR}/")


def create_test_image(path: str):
    from PIL import Image, ImageDraw

    img = Image.new('RGB', (400, 300), color='lightgray')
    draw = ImageDraw.Draw(img)

    draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
    draw.rectangle([200, 50, 300, 150], fill='green', outline='black')
    draw.rectangle([50, 180, 150, 280], fill='blue', outline='black')
    draw.rectangle([200, 180, 300, 280], fill='yellow', outline='black')

    draw.text((160, 10), "Test Image", fill='black')

    img.save(path)


if __name__ == "__main__":
    main()