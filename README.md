# Camera Intrinsics Utilities

This module provides a `CameraIntrinsics` class in the `core.py` script to manage and manipulate camera intrinsic parameters for a pinhole camera model. It supports both pixel-based and normalized intrinsics, along with a variety of transformations such as rescaling, padding, undistorting points, and conversion between pixel and normalized coordinates.

## Features

- **Initialization with Normalized or Pixel Units:**  
  Initialize intrinsic parameters in either normalized units (range [0,1]) or pixel units.
  
- **Intrinsic Matrix Computation:**  
  Automatically computes the 3×3 intrinsic matrix based on focal lengths, principal point, and skew.

- **In-place and Immutable Operations:**  
  Methods like `pad()` and `rescale()` support both immutable (returning new instances) and in-place updates using an optional parameter (`inplace=True`).

- **Coordinate Conversions:**  
  Convert between pixel coordinates and normalized image coordinates.

- **3D Point Projection and Unprojection:**  
  Project 3D camera-space points to 2D image pixels and unproject pixel coordinates to 3D rays.

- **Undistortion:**  
  Iterative undistortion of pixel coordinates given distortion coefficients.

## Installation

Ensure you have [Python 3](https://www.python.org/) installed along with [NumPy](https://numpy.org/).

You can install NumPy using pip:

```bash
pip install numpy
```

Copy the `camera_intrinsics_utilities.py` module into your project directory.

## Usage Example

Below is an example of how to use the `CameraIntrinsics` class:

```python
from camera_intrinsics_utilities import CameraIntrinsics

# Create an instance with normalized intrinsics (fx, fy provided as normalized values)
FX = 0.5
FY = 0.5
original_size = (640, 360)
CI = CameraIntrinsics(fx=FX, fy=FY, dimensions=original_size, normalized=True)

# Display the intrinsic matrix
print("Intrinsic Matrix:")
print(CI.as_matrix())

# Use immutable padding (returns a new instance)
CI_padded = CI.pad(140, 140)
print("Padded Intrinsic Matrix (Immutable):")
print(CI_padded.as_matrix())

# Use in-place update for padding
CI.pad(140, 140, inplace=True)
print("Intrinsic Matrix after in-place padding:")
print(CI.as_matrix())

# Use in-place update for rescaling
CI.rescale(800, 450, inplace=True)
print("Intrinsic Matrix after in-place rescaling:")
print(CI.as_matrix())
```

## API Overview

### Class: `CameraIntrinsics`

- **Constructor:**
  - `CameraIntrinsics(fx, fy, dimensions, cx=None, cy=None, skew=0.0, distortion=(0.0, 0.0, 0.0, 0.0, 0.0), normalized=False)`

- **Methods:**
  - `as_matrix()`: Returns the 3×3 intrinsic matrix.
  - `update()`: Recomputes the intrinsic matrix if parameters have been modified.
  - `rescale(new_width, new_height, inplace=False)`: Rescales the camera parameters to new dimensions.
  - `pad(pad_left, pad_right, inplace=False)`: Pads the image width, adjusting the principal point.
  - `inverse_K()`: Returns the inverse of the intrinsic matrix.
  - `pixel_to_normalized(points)`: Converts pixel coordinates to normalized coordinates.
  - `normalized_to_pixel(points)`: Converts normalized coordinates back to pixel coordinates.
  - `project_points(points_3d)`: Projects 3D points to image pixels using the pinhole camera model.
  - `unproject_pixel_to_ray(points)`: Converts pixel coordinates to a 3D ray.
  - `undistort_points(points, iterations=5)`: Undistorts pixel coordinates based on the distortion coefficients.
  - `fov_x()`: Computes the horizontal field-of-view (degrees).
  - `fov_y()`: Computes the vertical field-of-view (degrees).

## License

This code is provided "as-is", without any warranty. Feel free to use or modify it as needed.

## Contributing

If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request.
