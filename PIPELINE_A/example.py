"""
Example usage of the DroneLandmineSystem library
"""

from drone_landmine_detector import (
    DroneLandmineSystem,
    DroneImageStitcher,
    LandmineDetector,
    PathFinder
)


# =============================================================================
# EXAMPLE 1: Complete Pipeline (Recommended)
# =============================================================================

def run_complete_pipeline():
    """Run the entire pipeline in one go."""

    # Configuration
    MODEL_PATH = r"E:\dataset_img\yolo_dataset\runs\detect\train10\weights\best.pt"
    CONFIDENCE_THRESHOLD = 0.3

    DRONE_VIDEOS = [
        "./drone_1.mp4",
        "./drone_2.mp4",
        "./drone_3.mp4",
        "./drone_4.mp4"
    ]

    START_POINT = (184, 210)
    END_POINT = (1721, 643)

    # Initialize system
    system = DroneLandmineSystem(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # Run complete pipeline
    results = system.run_complete_pipeline(
        video_paths=DRONE_VIDEOS,
        start_point=START_POINT,
        end_point=END_POINT,
        frame_interval=10,
        grid_resolution=10,
        safety_margin=30
    )

    # Access results
    print(f"\nPipeline completed!")
    print(f"Stitching successful: {results['stitching_success']}")
    print(f"Detections found: {results['num_detections']}")
    print(f"Path found: {results['path_found']}")


# =============================================================================
# EXAMPLE 2: Step-by-Step Approach
# =============================================================================

def run_step_by_step():
    """Run each step individually with more control."""

    # Configuration
    MODEL_PATH = r"E:\dataset_img\yolo_dataset\runs\detect\train10\weights\best.pt"
    DRONE_VIDEOS = [
        "./drone_1.mp4",
        "./drone_2.mp4",
        "./drone_3.mp4",
        "./drone_4.mp4"
    ]

    # Step 1: Stitch videos
    print("Step 1: Stitching videos...")
    stitcher = DroneImageStitcher(
        frame_interval=10,
        enhance_image=True
    )
    stitched_image = stitcher.stitch(
        video_paths=DRONE_VIDEOS,
        output_path="my_stitched_image.tiff"
    )

    if stitched_image is None:
        print("Stitching failed!")
        return

    # Step 2: Detect landmines
    print("\nStep 2: Detecting landmines...")
    detector = LandmineDetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.3
    )
    detections = detector.detect("my_stitched_image.tiff")
    detector.print_summary(detections)
    detector.visualize_detections(
        "my_stitched_image.tiff",
        detections,
        "my_detections.jpg"
    )

    # Step 3: Find safe path
    print("\nStep 3: Finding safe path...")
    import cv2
    image = cv2.imread("my_stitched_image.tiff")

    pathfinder = PathFinder(
        image_width=image.shape[1],
        image_height=image.shape[0],
        grid_resolution=10
    )
    pathfinder.add_obstacles(detections, safety_margin=30)

    start = (184, 210)
    end = (1721, 643)
    path = pathfinder.find_path(start, end)

    if path:
        print(f"Found path with {len(path)} waypoints")
        pathfinder.visualize_path(
            "my_stitched_image.tiff",
            path,
            detections,
            "my_safe_path.jpg"
        )


# =============================================================================
# EXAMPLE 3: Using Only Detection on Pre-Stitched Image
# =============================================================================

def detect_on_existing_image():
    """Use only the detection module on an existing image."""

    MODEL_PATH = r"E:\dataset_img\yolo_dataset\runs\detect\train10\weights\best.pt"
    IMAGE_PATH = "./existing_panorama.tiff"

    # Initialize detector
    detector = LandmineDetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.25
    )

    # Detect landmines
    detections = detector.detect(IMAGE_PATH)

    # Print statistics
    detector.print_summary(detections)

    # Visualize results
    detector.visualize_detections(
        IMAGE_PATH,
        detections,
        "detections_output.jpg"
    )

    return detections


# =============================================================================
# EXAMPLE 4: Custom Path Finding with Multiple Start/End Points
# =============================================================================

def find_multiple_paths():
    """Find multiple paths on the same image."""

    MODEL_PATH = r"E:\dataset_img\yolo_dataset\runs\detect\train10\weights\best.pt"
    IMAGE_PATH = "./stitched_drone_image.tiff"

    # Initialize system
    system = DroneLandmineSystem(
        model_path=MODEL_PATH,
        confidence_threshold=0.3
    )

    # Detect landmines once
    detections = system.detect_landmines(IMAGE_PATH)

    # Define multiple routes
    routes = [
        {"name": "Route A", "start": (100, 100), "end": (1500, 800)},
        {"name": "Route B", "start": (200, 300), "end": (1600, 600)},
        {"name": "Route C", "start": (300, 200), "end": (1400, 900)},
    ]

    # Find paths for each route
    import cv2
    for i, route in enumerate(routes):
        print(f"\nFinding path for {route['name']}...")
        path = system.find_safe_path(
            start_point=route['start'],
            end_point=route['end'],
            grid_resolution=10,
            safety_margin=30,
            image_path=IMAGE_PATH
        )

        if path:
            system.pathfinder.visualize_path(
                IMAGE_PATH,
                path,
                detections,
                f"path_{route['name'].replace(' ', '_')}.jpg"
            )


# =============================================================================
# EXAMPLE 5: Advanced Configuration
# =============================================================================

def advanced_usage():
    """Example with advanced configuration options."""

    # Initialize with custom parameters
    stitcher = DroneImageStitcher(
        frame_interval=5,  # More frames for better quality
        max_dimension=6000,  # Higher resolution
        enhance_image=True,
        border_padding=30  # More padding
    )

    # Stitch with custom output
    result = stitcher.stitch(
        video_paths=["./drone_1.mp4", "./drone_2.mp4"],
        output_path="high_quality_stitch.tiff"
    )

    # Detect with GPU
    detector = LandmineDetector(
        model_path="./best.pt",
        confidence_threshold=0.4,  # Higher threshold
        device="cuda:0"  # Use GPU
    )

    # Detect with custom SAHI parameters
    detections = detector.detect(
        "high_quality_stitch.tiff",
        slice_height=800,  # Larger slices
        slice_width=800,
        overlap_ratio=0.3  # More overlap
    )

    # Pathfinding with fine grid
    import cv2
    image = cv2.imread("high_quality_stitch.tiff")

    pathfinder = PathFinder(
        image_width=image.shape[1],
        image_height=image.shape[0],
        grid_resolution=5  # Finer grid (smaller cells)
    )
    pathfinder.add_obstacles(
        detections,
        safety_margin=50  # Larger safety margin
    )

    path = pathfinder.find_path((100, 100), (1500, 800))


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    # Choose which example to run

    # Option 1: Complete pipeline (easiest)
    run_complete_pipeline()

    # Option 2: Step by step (more control)
    # run_step_by_step()

    # Option 3: Detection only
    # detect_on_existing_image()

    # Option 4: Multiple paths
    # find_multiple_paths()

    # Option 5: Advanced configuration
    # advanced_usage()