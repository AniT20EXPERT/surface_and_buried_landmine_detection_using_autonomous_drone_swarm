"""
Drone Landmine Detection Library
A comprehensive library for stitching drone videos, detecting landmines using YOLO+SAHI,
and finding safe paths using A* pathfinding algorithm.
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Tuple, Optional
import heapq
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class DroneImageStitcher:
    """Handles stitching of multiple drone video feeds into a single panorama."""

    def __init__(self, frame_interval: int = 10, max_dimension: int = 5000,
                 enhance_image: bool = True, border_padding: int = 20):
        """
        Initialize the DroneImageStitcher.

        Args:
            frame_interval: Extract every nth frame from videos
            max_dimension: Maximum dimension for images to prevent memory issues
            enhance_image: Whether to enhance brightness/contrast
            border_padding: Padding for border cropping
        """
        self.frame_interval = frame_interval
        self.max_dimension = max_dimension
        self.enhance_image = enhance_image
        self.border_padding = border_padding
        cv2.setNumThreads(4)

    def capture_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from a video file."""
        if not video_path or video_path == "path/to/drone1.mp4":
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return []

        images = []
        frame_count = 0
        last_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            last_frame = frame.copy()

            if frame_count == 0 or frame_count % self.frame_interval == 0:
                images.append(frame)

            frame_count += 1

        cap.release()

        if last_frame is not None and len(images) > 0:
            if not np.array_equal(images[-1], last_frame):
                images.append(last_frame)
                print(f"Added last frame to ensure complete coverage")

        print(f"Extracted {len(images)} frames from {video_path}")
        return images

    def enhance_brightness_contrast(self, image: np.ndarray,
                                    brightness: int = 20,
                                    contrast: float = 1.15) -> np.ndarray:
        """Enhance image brightness, contrast, and sharpness."""
        enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

        sharpening_kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])

        sharpened = cv2.filter2D(enhanced, -1, sharpening_kernel)
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

        return result

    def resize_if_too_large(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """Resize images if they exceed max_dimension."""
        if not images:
            return images, 1.0

        h, w = images[0].shape[:2]
        max_dim = max(h, w)

        if max_dim > self.max_dimension:
            scale_factor = self.max_dimension / max_dim
            print(f"Resizing images by {scale_factor:.3f} to prevent memory issues")

            resized_images = []
            for img in images:
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                resized_images.append(resized)

            return resized_images, scale_factor

        return images, 1.0

    def crop_black_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove black borders from stitched image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            padding = self.border_padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            cropped = image[y:y + h, x:x + w]
            print(f"Cropped from {image.shape[1]}x{image.shape[0]} to {cropped.shape[1]}x{cropped.shape[0]}")
            return cropped

        return image

    def stitch(self, video_paths: List[str], output_path: str = "stitched_drone_image.tiff") -> Optional[np.ndarray]:
        """
        Stitch multiple drone videos into a single panorama.

        Args:
            video_paths: List of paths to drone video files
            output_path: Path to save the stitched image

        Returns:
            Stitched image as numpy array, or None if stitching failed
        """
        all_images = []

        for video_path in video_paths:
            frames = self.capture_frames(video_path)
            if frames:
                all_images.extend(frames)

        if not all_images:
            print("No images found! Check your video paths.")
            return None

        print(f"\nTotal images for stitching: {len(all_images)}")
        print(f"Original image size: {all_images[0].shape[1]}x{all_images[0].shape[0]}")

        all_images, scale_factor = self.resize_if_too_large(all_images)

        if scale_factor < 1.0:
            print(f"Images resized by factor {scale_factor:.3f}")

        print(f"\nStarting stitching process...")

        try:
            stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
            print("Using SCANS mode for aerial footage")
        except:
            stitcher = cv2.Stitcher_create()
            print("Using default stitching mode")

        try:
            stitcher.setRegistrationResol(0.6)
            stitcher.setSeamEstimationResol(0.1)
            stitcher.setCompositingResol(-1)
            stitcher.setPanoConfidenceThresh(0.8)
        except:
            print("Warning: Could not set advanced stitcher parameters")

        status, stitched = stitcher.stitch(all_images)

        if status == cv2.Stitcher_OK:
            print(f"Raw stitched size: {stitched.shape[1]}x{stitched.shape[0]}")
            print("\nPost-processing...")

            stitched = self.crop_black_borders(stitched)

            if self.enhance_image:
                print("Enhancing brightness and contrast...")
                stitched = self.enhance_brightness_contrast(stitched)

            print(f"Final image size: {stitched.shape[1]}x{stitched.shape[0]}")

            success = cv2.imwrite(output_path, stitched)
            if success:
                print(f"Stitching successful! Saved to {output_path}")

            return stitched
        else:
            error_messages = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed - check image overlap",
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed"
            }
            error_msg = error_messages.get(status, f"Unknown error (code: {status})")
            print(f"Stitching failed: {error_msg}")
            return None


class LandmineDetector:
    """Handles landmine detection using YOLO and SAHI."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.3,
                 device: str = "cpu"):
        """
        Initialize the LandmineDetector.

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu' or 'cuda:0')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.detection_model = None

    def load_model(self):
        """Load the YOLO detection model."""
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )
        print("Model loaded successfully")

    def detect(self, image_path: str, slice_height: int = 640,
               slice_width: int = 640, overlap_ratio: float = 0.25) -> List[Dict]:
        """
        Detect landmines in an image using SAHI.

        Args:
            image_path: Path to the image file
            slice_height: Height of each slice for SAHI
            slice_width: Width of each slice for SAHI
            overlap_ratio: Overlap ratio between slices

        Returns:
            List of detection dictionaries
        """
        if self.detection_model is None:
            self.load_model()

        result = get_sliced_prediction(
            image_path,
            self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )

        return self._extract_detection_data(result.object_prediction_list)

    def _extract_detection_data(self, object_prediction_list) -> List[Dict]:
        """Extract structured data from SAHI predictions."""
        detections = []

        for i, prediction in enumerate(object_prediction_list):
            bbox = prediction.bbox

            x1, y1 = bbox.minx, bbox.miny
            x2, y2 = bbox.maxx, bbox.maxy

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            width = x2 - x1
            height = y2 - y1

            detection = {
                'id': i + 1,
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'center': {
                    'x': float(center_x),
                    'y': float(center_y)
                },
                'dimensions': {
                    'width': float(width),
                    'height': float(height),
                    'area': float(width * height)
                },
                'confidence': float(prediction.score.value),
                'class': {
                    'id': int(prediction.category.id),
                    'name': str(prediction.category.name)
                }
            }

            detections.append(detection)

        return detections

    def print_summary(self, detections: List[Dict]):
        """Print detection statistics."""
        print(f"Total detections: {len(detections)}")

        if not detections:
            print("No landmines detected!")
            return

        confidences = [d['confidence'] for d in detections]
        areas = [d['dimensions']['area'] for d in detections]

        print(f"Avg confidence: {np.mean(confidences):.3f}")
        print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"Avg detection area: {np.mean(areas):.1f} pixels²")

    def visualize_detections(self, image_path: str, detections: List[Dict],
                             output_path: str = "detections.jpg"):
        """Draw bounding boxes on the image."""
        image = cv2.imread(image_path)
        print(f"Plotting {len(detections)} bounding boxes")

        for detection in detections:
            x1 = int(detection['bbox']['x1'])
            y1 = int(detection['bbox']['y1'])
            x2 = int(detection['bbox']['x2'])
            y2 = int(detection['bbox']['y2'])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(output_path, image)
        print(f"Saved visualization to: {output_path}")


class PathFinder:
    """A* pathfinding algorithm to find safe paths avoiding landmines."""

    def __init__(self, image_width: int, image_height: int,
                 grid_resolution: int = 10):
        """
        Initialize the PathFinder.

        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            grid_resolution: Size of each grid cell in pixels
        """
        self.width = image_width
        self.height = image_height
        self.grid_resolution = grid_resolution
        self.grid_width = image_width // grid_resolution
        self.grid_height = image_height // grid_resolution
        self.obstacle_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)

    def add_obstacles(self, detections: List[Dict], safety_margin: int = 50):
        """
        Mark detected landmines as obstacles on the grid.

        Args:
            detections: List of detection dictionaries
            safety_margin: Extra margin around each landmine in pixels
        """
        for detection in detections:
            bbox = detection['bbox']

            x1 = max(0, int((bbox['x1'] - safety_margin) // self.grid_resolution))
            y1 = max(0, int((bbox['y1'] - safety_margin) // self.grid_resolution))
            x2 = min(self.grid_width, int((bbox['x2'] + safety_margin) // self.grid_resolution))
            y2 = min(self.grid_height, int((bbox['y2'] + safety_margin) // self.grid_resolution))

            self.obstacle_grid[y1:y2 + 1, x1:x2 + 1] = True

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-directional movement)."""
        x, y = node
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                if (0 <= nx < self.grid_width and
                        0 <= ny < self.grid_height and
                        not self.obstacle_grid[ny, nx]):
                    neighbors.append((nx, ny))

        return neighbors

    def find_path(self, start_pixel: Tuple[int, int],
                  end_pixel: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find a safe path from start to end using A* algorithm.

        Args:
            start_pixel: Starting point (x, y) in pixels
            end_pixel: Ending point (x, y) in pixels

        Returns:
            List of waypoints (x, y) in pixels, or empty list if no path found
        """
        start_grid = (start_pixel[0] // self.grid_resolution,
                      start_pixel[1] // self.grid_resolution)
        end_grid = (end_pixel[0] // self.grid_resolution,
                    end_pixel[1] // self.grid_resolution)

        if (self.obstacle_grid[start_grid[1], start_grid[0]] or
                self.obstacle_grid[end_grid[1], end_grid[0]]):
            print("Start or end point is blocked!")
            return []

        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, end_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == end_grid:
                path = []
                while current in came_from:
                    pixel_coord = (current[0] * self.grid_resolution + self.grid_resolution // 2,
                                   current[1] * self.grid_resolution + self.grid_resolution // 2)
                    path.append(pixel_coord)
                    current = came_from[current]

                start_pixel_center = (start_grid[0] * self.grid_resolution + self.grid_resolution // 2,
                                      start_grid[1] * self.grid_resolution + self.grid_resolution // 2)
                path.append(start_pixel_center)

                return path[::-1]

            for neighbor in self._get_neighbors(current):
                dx, dy = abs(neighbor[0] - current[0]), abs(neighbor[1] - current[1])
                move_cost = 1.414 if dx + dy == 2 else 1

                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("No path found!")
        return []

    def visualize_path(self, image_path: str, path: List[Tuple[int, int]],
                       detections: List[Dict], output_path: str = "safe_path.jpg"):
        """
        Visualize the safe path on the image.

        Args:
            image_path: Path to the input image
            path: List of waypoints
            detections: List of detection dictionaries
            output_path: Path to save the visualization
        """
        image = cv2.imread(image_path)

        for det in detections:
            bbox = det['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"{det['confidence']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

        if len(path) > 1:
            for i in range(len(path) - 1):
                cv2.line(image, path[i], path[i + 1], (0, 255, 0), 3)

            cv2.circle(image, path[0], 8, (255, 0, 0), -1)
            cv2.circle(image, path[-1], 8, (0, 255, 255), -1)

        cv2.imwrite(output_path, image)
        print(f"Saved path visualization to: {output_path}")


class DroneLandmineSystem:
    """
    Complete system integrating stitching, detection, and pathfinding.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.3):
        """
        Initialize the complete drone landmine detection system.

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.stitcher = None
        self.detector = LandmineDetector(model_path, confidence_threshold)
        self.pathfinder = None
        self.detections = []
        self.stitched_image_path = None

    def stitch_videos(self, video_paths: List[str], output_path: str = "stitched_drone_image.tiff",
                      frame_interval: int = 10) -> bool:
        """Stitch multiple drone videos."""
        self.stitcher = DroneImageStitcher(frame_interval=frame_interval)
        result = self.stitcher.stitch(video_paths, output_path)

        if result is not None:
            self.stitched_image_path = output_path
            return True
        return False

    def detect_landmines(self, image_path: Optional[str] = None) -> List[Dict]:
        """Detect landmines in the image."""
        if image_path is None:
            image_path = self.stitched_image_path

        if image_path is None:
            raise ValueError("No image path provided and no stitched image available")

        self.detections = self.detector.detect(image_path)
        self.detector.print_summary(self.detections)
        return self.detections

    def find_safe_path(self, start_point: Tuple[int, int], end_point: Tuple[int, int],
                       grid_resolution: int = 10, safety_margin: int = 30,
                       image_path: Optional[str] = None) -> List[Tuple[int, int]]:
        """Find a safe path avoiding detected landmines."""
        if image_path is None:
            image_path = self.stitched_image_path

        if image_path is None:
            raise ValueError("No image path provided and no stitched image available")

        image = cv2.imread(image_path)

        self.pathfinder = PathFinder(image.shape[1], image.shape[0], grid_resolution)
        self.pathfinder.add_obstacles(self.detections, safety_margin)

        path = self.pathfinder.find_path(start_point, end_point)

        if path:
            print(f"Safe path found with {len(path)} waypoints")

        return path

    def run_complete_pipeline(self, video_paths: List[str],
                              start_point: Tuple[int, int],
                              end_point: Tuple[int, int],
                              frame_interval: int = 10,
                              grid_resolution: int = 10,
                              safety_margin: int = 30) -> Dict:
        """
        Run the complete pipeline: stitch -> detect -> pathfind.

        Returns:
            Dictionary with results and timings
        """
        results = {}

        # Step 1: Stitching
        print("=" * 50)
        print("STEP 1: STITCHING VIDEOS")
        print("=" * 50)
        start_time = time.time()
        success = self.stitch_videos(video_paths, frame_interval=frame_interval)
        results['stitching_time'] = time.time() - start_time
        results['stitching_success'] = success

        if not success:
            print("Stitching failed!")
            return results

        # Step 2: Detection
        print("\n" + "=" * 50)
        print("STEP 2: DETECTING LANDMINES")
        print("=" * 50)
        start_time = time.time()
        detections = self.detect_landmines()
        results['detection_time'] = time.time() - start_time
        results['num_detections'] = len(detections)

        self.detector.visualize_detections(self.stitched_image_path,
                                           detections, "detections.jpg")

        # Step 3: Pathfinding
        print("\n" + "=" * 50)
        print("STEP 3: FINDING SAFE PATH")
        print("=" * 50)
        start_time = time.time()
        path = self.find_safe_path(start_point, end_point,
                                   grid_resolution, safety_margin)
        results['pathfinding_time'] = time.time() - start_time
        results['path_length'] = len(path)
        results['path_found'] = len(path) > 0

        if path:
            self.pathfinder.visualize_path(self.stitched_image_path,
                                           path, detections, "safe_path.jpg")

        # Print summary
        print("\n" + "=" * 50)
        print("PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Stitching time: {results['stitching_time']:.1f}s")
        print(f"Detection time: {results['detection_time']:.1f}s")
        print(f"Pathfinding time: {results['pathfinding_time']:.1f}s")
        print(
            f"Total time: {sum([results['stitching_time'], results['detection_time'], results['pathfinding_time']]):.1f}s")
        print(f"Detections found: {results['num_detections']}")
        print(f"Safe path: {'Found' if results['path_found'] else 'Not found'}")

        return results