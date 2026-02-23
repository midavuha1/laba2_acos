import threading
import time
import os
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from PIL import Image, ImageDraw
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger()


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
                else:
                    new_rgb = (r, g, b)

                new_pixels.append(new_rgb)

        return new_pixels


class Producer(threading.Thread):
    def __init__(self, task_queue: BlockingQueue, image_files: List[str], output_dir: str, filter_type: FilterType,
                 poison_pills_count: int):
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
                logger.info(f"Producer added task: {base_name}")

        except Exception as e:
            logger.error(f"Error in Producer: {e}")

        finally:
            for i in range(self.poison_pills_count):
                self.task_queue.put(None)
            logger.info(f"Producer finished. Added {self.processed_count} tasks")


class Consumer(threading.Thread):
    def __init__(self, consumer_id: int, task_queue: BlockingQueue, results_queue: BlockingQueue):
        super().__init__()
        self.consumer_id = consumer_id
        self.task_queue = task_queue
        self.results_queue = results_queue
        self.processed_count = 0
        self.running = True

    def run(self):
        logger.info(f"Consumer {self.consumer_id} launched")

        while True:
            try:
                task = self.task_queue.get()

                if task is None:
                    logger.info(f"Consumer {self.consumer_id} received poison pill, stopping")
                    break

                logger.info(f"Consumer {self.consumer_id} processing: {os.path.basename(task.input_path)}")

                start_time = time.time()
                success, error_msg = self.process_image(task)
                proc_time = time.time() - start_time

                if success:
                    logger.info(
                        f"Consumer {self.consumer_id} completed: {os.path.basename(task.input_path)} in {proc_time:.2f}s")
                else:
                    logger.error(
                        f"Consumer {self.consumer_id} error processing {os.path.basename(task.input_path)}: {error_msg}")

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
                logger.error(f"Error in Consumer-{self.consumer_id}: {e}")
                continue

    def process_image(self, task: ImageTask):
        try:
            with Image.open(task.input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                pixels = list(img.getdata())
                width, height = img.size

                if task.filter_type == FilterType.BLUR:
                    new_pixels = self.apply_gaussian_blur(pixels, width, height, radius=2.0)
                else:
                    new_pixels = ImageProcessor.apply_filter(
                        pixels, width, height, task.filter_type
                    )

                new_img = Image.new('RGB', (width, height))
                new_img.putdata(new_pixels)

                new_img.save(task.output_path, 'PNG', optimize=True)

                return True, ""

        except Exception as e:
            return False, str(e)

    def apply_gaussian_blur(self, pixels, width, height, radius=2.0):
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
        max_wait = 30
        start_time = time.time()

        while completed < self.total_tasks:
            if time.time() - start_time > max_wait:
                logger.warning(f"Results collector timeout. Collected {completed}/{self.total_tasks} results")
                break

            try:
                result = self.results_queue.get()
                self.results.append(result)
                completed += 1
                start_time = time.time()

            except Exception:
                continue

        logger.info(f"Results collector finished. Collected {len(self.results)} results")


def find_images(directory: str) -> List[str]:
    extensions = {'.jpg', '.jpeg', '.png'}
    images = []

    if not os.path.exists(directory):
        return images

    for file in os.listdir(directory):
        ext = os.path.splitext(file)[1].lower()
        if ext in extensions:
            images.append(os.path.join(directory, file))

    return sorted(images)


def create_test_images(count: int, directory: str):
    os.makedirs(directory, exist_ok=True)

    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']

    for i in range(count):
        img_path = os.path.join(directory, f"test_image_{i + 1}.png")

        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)

        color = colors[i % len(colors)]
        draw.rectangle([100, 100, 700, 500], fill=color, outline='black', width=5)
        draw.text((350, 280), f"Test Image {i + 1}", fill='black')

        img.save(img_path)
        logger.info(f"Created test image: {img_path}")


def print_results_summary(results: List[ImageResult]):
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    print(f"\nTotal tasks: {len(results)}")
    print(f"Successfully: {successful}")
    print(f"Errors: {failed}")

    if failed > 0:
        print("\nErrors:")
        for result in results:
            if not result.success:
                print(f"  - {os.path.basename(result.output_path)}: {result.error_message}")


def main():
    INPUT_DIR = "./input_images"
    OUTPUT_DIR = "./output_images"

    print("\nAvailable filters:")
    filters = list(FilterType)
    for i, filter_type in enumerate(filters, 1):
        print(f"  {i}. {filter_type.value}")

    try:
        choice = int(input("\nSelect a filter (1-5): ").strip())
        selected_filter = filters[choice - 1]
    except (ValueError, IndexError):
        selected_filter = FilterType.INVERT
        print(f"Using default filter: {selected_filter.value}")

    print(f"Selected filter: {selected_filter.value}")

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = find_images(INPUT_DIR)

    if not images:
        print("\nNo images found in input folder. Creating test images...")
        create_test_images(5, INPUT_DIR)
        images = find_images(INPUT_DIR)

    print(f"\nFound {len(images)} images")

    try:
        num_consumers = int(input(f"\nHow many consumers to use? (1-10, default 4): ") or 4)
        num_consumers = max(1, min(num_consumers, 10))
    except ValueError:
        num_consumers = 4

    print(f"Using {num_consumers} consumers")

    task_queue = BlockingQueue(maxsize=num_consumers * 2)
    results_queue = BlockingQueue()

    producer = Producer(task_queue, images, OUTPUT_DIR, selected_filter, num_consumers)

    consumers = []
    for i in range(num_consumers):
        consumer = Consumer(i, task_queue, results_queue)
        consumers.append(consumer)

    collector = ResultsCollector(results_queue, len(images))

    start_time = time.time()

    collector.start()
    for consumer in consumers:
        consumer.start()
    producer.start()

    producer.join()
    logger.info("Producer finished")

    for consumer in consumers:
        consumer.join()
    logger.info("All consumers finished")

    collector.join()
    logger.info("Collector finished")

    total_time = time.time() - start_time

    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print_results_summary(collector.results)
    print(f"Results saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        logger.exception("Critical error in main")
