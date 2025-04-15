import numpy as np
from typing import Union, Sequence, Tuple

class CameraIntrinsics:
    def __init__(self,
                 fx: float,
                 fy: float,
                 dimensions: Tuple[int, int],
                 cx: float = None,
                 cy: float = None,
                 skew: float = 0.0,
                 distortion: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
                 normalized: bool = False):
        """
        Initializes the intrinsic parameters for a pinhole camera.
        
        If normalized is True, the provided fx, fy, cx, cy, and skew are assumed to be in
        normalized units (relative to the image dimensions) and will be converted to pixel units.
        
        Args:
          fx (float): Focal length (in pixels or normalized units if normalized=True) along x.
          fy (float): Focal length (in pixels or normalized units if normalized=True) along y.
          dimensions (Tuple[int, int]): A tuple (width, height) of the image dimensions.
          cx (float, optional): Principal point x-coordinate; defaults to width/2 (or 0.5 if normalized=True).
          cy (float, optional): Principal point y-coordinate; defaults to height/2 (or 0.5 if normalized=True).
          skew (float, optional): Skew parameter between image axes (default 0).
          distortion (tuple, optional): Distortion coefficients (k1, k2, p1, p2, k3).
          normalized (bool, optional): If True, fx, fy, cx, cy, and skew are given in normalized units.
                                       Defaults to False.
        """
        self.width, self.height = dimensions
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")
        
        if normalized:
            # Convert normalized intrinsics to pixel units.
            self.fx = fx * self.width
            self.fy = fy * self.height
            self.cx = cx * self.width if cx is not None else self.width / 2.0
            self.cy = cy * self.height if cy is not None else self.height / 2.0
            # Since skew is an x-direction property, we scale by the image width.
            self.skew = skew * self.width
        else:
            self.fx = fx
            self.fy = fy
            self.cx = cx if cx is not None else self.width / 2.0
            self.cy = cy if cy is not None else self.height / 2.0
            self.skew = skew
        
        self.distortion = distortion
        
        self.update()  # Build the intrinsic matrix

    def update(self):
        """
        Recompute the intrinsic matrix from current parameters.
        Use this method if you modify any of the intrinsic parameters in place.
        """
        self.K_matrix = np.array([[self.fx, self.skew, self.cx],
                                  [0.0,     self.fy,    self.cy],
                                  [0.0,     0.0,        1.0]], dtype=float)
        return self

    def K(self) -> np.ndarray:
        """
        Returns the 3x3 intrinsic matrix.
        """
        return self.K_matrix
    
    def as_matrix(self) -> np.ndarray:
        return self.K()
    
    def rescale(self, new_width: int, new_height: int, inplace: bool = False) -> "CameraIntrinsics":
        """
        Rescales the intrinsic parameters proportionally to the new image dimensions.
        
        If inplace is True, updates the current instance; otherwise returns a new instance.
        """
        if new_width <= 0 or new_height <= 0:
            raise ValueError("New image dimensions must be positive")
            
        scale_x = new_width / self.width
        scale_y = new_height / self.height

        if inplace:
            self.fx *= scale_x
            self.fy *= scale_y
            self.cx *= scale_x
            self.cy *= scale_y
            self.skew *= scale_x   # Typically skew scales like fx.
            self.width = new_width
            self.height = new_height
            self.update()
            return self
        else:
            return CameraIntrinsics(
                fx=self.fx * scale_x,
                fy=self.fy * scale_y,
                dimensions=(new_width, new_height),
                cx=self.cx * scale_x,
                cy=self.cy * scale_y,
                skew=self.skew * scale_x,
                distortion=self.distortion,
                normalized=False  # Stored in pixel units.
            )
    
    def pad(self, pad_left: int, pad_right: int, inplace: bool = False) -> "CameraIntrinsics":
        """
        Pads the image width. The height remains unchanged and the horizontal principal point shifts by pad_left.
        
        If inplace is True, updates the current instance; otherwise returns a new instance.
        """
        if pad_left < 0 or pad_right < 0:
            raise ValueError("Padding values must be non-negative")
            
        new_width = self.width + pad_left + pad_right

        if inplace:
            self.cx += pad_left
            self.width = new_width
            self.update()
            return self
        else:
            return CameraIntrinsics(
                fx=self.fx,
                fy=self.fy,
                dimensions=(new_width, self.height),
                cx=self.cx + pad_left,
                cy=self.cy,
                skew=self.skew,
                distortion=self.distortion,
                normalized=False
            )
    
    def inverse_K(self) -> np.ndarray:
        """
        Returns the inverse of the intrinsic matrix.
        """
        return np.linalg.inv(self.K_matrix)
    
    def pixel_to_normalized(self, points: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """
        Converts pixel coordinates to normalized image coordinates.
        For cameras with skew:
          x_norm = (u - cx - skew*(v - cy)) / fx,
          y_norm = (v - cy) / fy.
          
        Args:
          points: A 2-element sequence [u, v] or an (N,2) numpy array.
          
        Returns:
          A numpy array of normalized coordinates.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            if pts.shape[0] != 2:
                raise ValueError("Input must have 2 elements: (u, v).")
            u, v = pts
            x_norm = (u - self.cx - self.skew * (v - self.cy)) / self.fx
            y_norm = (v - self.cy) / self.fy
            return np.array([x_norm, y_norm])
        elif pts.ndim == 2:
            if pts.shape[1] != 2:
                raise ValueError("Each point must have 2 coordinates (u, v).")
            u = pts[:, 0]
            v = pts[:, 1]
            x_norm = (u - self.cx - self.skew * (v - self.cy)) / self.fx
            y_norm = (v - self.cy) / self.fy
            return np.column_stack((x_norm, y_norm))
        else:
            raise ValueError("Input must be a 2-element sequence or an (N,2) array.")
    
    def normalized_to_pixel(self, points: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """
        Converts normalized image coordinates back to pixel coordinates.
        For cameras with skew:
          u = fx*x + skew*y + cx,
          v = fy*y + cy.
          
        Args:
          points: A 2-element sequence or an (N,2) numpy array.
          
        Returns:
          A numpy array of pixel coordinates.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            if pts.shape[0] != 2:
                raise ValueError("Input must have 2 elements: (x, y).")
            x, y = pts
            u = self.fx * x + self.skew * y + self.cx
            v = self.fy * y + self.cy
            return np.array([u, v])
        elif pts.ndim == 2:
            if pts.shape[1] != 2:
                raise ValueError("Each point must have 2 coordinates (x, y).")
            x = pts[:, 0]
            y = pts[:, 1]
            u = self.fx * x + self.skew * y + self.cx
            v = self.fy * y + self.cy
            return np.column_stack((u, v))
        else:
            raise ValueError("Input must be a 2-element sequence or an (N,2) array.")
    
    def fov_x(self) -> float:
        """
        Computes the horizontal field-of-view (in degrees) using:
          fov_x = 2 * arctan((width/2) / fx)
        """
        return 2 * np.degrees(np.arctan(self.width / (2 * self.fx)))
    
    def fov_y(self) -> float:
        """
        Computes the vertical field-of-view (in degrees) using:
          fov_y = 2 * arctan((height/2) / fy)
        """
        return 2 * np.degrees(np.arctan(self.height / (2 * self.fy)))
    
    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Projects 3D camera-space points to 2D image pixels using the pinhole model.
        
        For each point (X, Y, Z):
            u = fx*(X/Z) + skew*(Y/Z) + cx,
            v = fy*(Y/Z) + cy.
            
        Args:
            points_3d: An (N,3) numpy array. All Z values must be nonzero.
            
        Returns:
            An (N,2) array of pixel coordinates.
        """
        if not (isinstance(points_3d, np.ndarray) and points_3d.ndim == 2 and points_3d.shape[1] == 3):
            raise ValueError("points_3d must be an (N,3) numpy array.")
        Z = points_3d[:, 2]
        if np.any(Z == 0):
            raise ValueError("All points must have non-zero Z coordinates.")
        x_norm = points_3d[:, 0] / Z
        y_norm = points_3d[:, 1] / Z
        u = self.fx * x_norm + self.skew * y_norm + self.cx
        v = self.fy * y_norm + self.cy
        return np.column_stack((u, v))
    
    def unproject_pixel_to_ray(self, points: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """
        Converts pixel coordinates to 3D camera-space rays.
        
        For a pixel (u, v), first the normalized coordinates are computed:
            x_norm = (u - cx - skew*(v - cy)) / fx,
            y_norm = (v - cy) / fy,
        and the method returns the ray direction [x_norm, y_norm, 1]. 
        (The ray is not normalized.)
        
        Args:
            points: A 2-element sequence or an (N,2) numpy array.
            
        Returns:
            An array representing the ray(s).
        """
        normalized = self.pixel_to_normalized(points)
        if normalized.ndim == 1:
            x, y = normalized
            return np.array([x, y, 1.0])
        else:
            ones = np.ones((normalized.shape[0], 1))
            return np.concatenate([normalized, ones], axis=1)
    
    def undistort_points(self, points: Union[Sequence[float], np.ndarray], iterations: int = 5) -> np.ndarray:
        """
        Undistorts pixel coordinates using the camera's distortion coefficients.
        The method converts from distorted pixel coordinates to normalized coordinates,
        iteratively inverts the distortion model, and converts the corrected coordinates
        back to pixel coordinates.
        
        The distortion model assumes:
            r^2 = x^2 + y^2
            radial = 1 + k1*r^2 + k2*r^4 + k3*r^6
            delta_x = 2*p1*x*y + p2*(r^2 + 2*x^2)
            delta_y = p1*(r^2 + 2*y^2) + 2*p2*x*y
            
        Args:
            points: A 2-element sequence or an (N,2) numpy array.
            iterations: Number of iterations for refinement (default 5).
            
        Returns:
            The undistorted pixel coordinates.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
            
        # Convert distorted pixel coordinates to normalized coordinates.
        u = pts[:, 0]
        v = pts[:, 1]
        x_d = (u - self.cx - self.skew * (v - self.cy)) / self.fx
        y_d = (v - self.cy) / self.fy
        
        # Initialize with the distorted normalized coordinates.
        x_u = x_d.copy()
        y_u = y_d.copy()
        
        k1, k2, p1, p2, k3 = self.distortion
        
        for _ in range(iterations):
            r2 = x_u**2 + y_u**2
            radial = 1 + k1 * r2 + k2 * (r2**2) + k3 * (r2**3)
            delta_x = 2 * p1 * x_u * y_u + p2 * (r2 + 2 * x_u**2)
            delta_y = p1 * (r2 + 2 * y_u**2) + 2 * p2 * x_u * y_u
            x_u = (x_d - delta_x) / radial
            y_u = (y_d - delta_y) / radial
        
        # Convert the undistorted normalized coordinates back to pixel coordinates.
        u_undist = self.fx * x_u + self.skew * y_u + self.cx
        v_undist = self.fy * y_u + self.cy
        undistorted = np.column_stack((u_undist, v_undist))
        return undistorted if undistorted.shape[0] > 1 else undistorted.flatten()

# Example usage:
# if __name__ == "__main__":
#     # Create an instance with non-normalized intrinsics.
#     CI = CameraIntrinsics(fx=0.5, fy=0.5, dimensions=(640, 360), normalized=True)
#     print("Initial Intrinsic Matrix:")
#     print(CI.as_matrix())
    
#     # Immutable pad (returns new instance)
#     CI_new = CI.pad(140, 140)
#     print("Immutable pad result:")
#     print(CI_new.as_matrix())
    
#     # In-place pad update:
#     CI.pad(140, 140, inplace=True)
#     print("After in-place pad update:")
#     print(CI.as_matrix())
    
#     # Similarly, you can use in-place update for rescale:
#     CI.rescale(800, 450, inplace=True)
#     print("After in-place rescale update:")
#     print(CI.as_matrix())
