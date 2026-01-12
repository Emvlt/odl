"""Surface and Points class"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from coordinate_systems import (CoordinateSystem, CoordinateSystemRegistry, CartesianSystem, SphericalSystem, CylindricalSystem)

class Surface(ABC):
    """Base class for surfaces with intrinsic coordinate system"""
    
    def __init__(self, coordinate_system: CoordinateSystem):
        self.coord_system = coordinate_system
    
    @abstractmethod
    def contains_point(self, point: 'Point') -> bool:
        """Check if a point lies on the surface"""
        pass
    
    @abstractmethod
    def project_to_surface(self, point: 'Point') -> 'Point':
        """Project a point onto the surface"""
        pass
    
    @abstractmethod
    def normal_at(self, point: 'Point') -> np.ndarray:
        """Get the normal vector at a point on the surface (in Cartesian)"""
        pass
    
    @abstractmethod
    def surface_distance(self, p1: 'Point', p2: 'Point') -> float:
        """Geodesic distance between two points along the surface"""
        pass
    
    def create_point(self, coords: np.ndarray) -> 'Point':
        """Create a point on this surface"""
        return Point(coords, self.coord_system, surface=self)

class Line(Surface):
    """Line with natural parametrization (t, 0, 0)"""
    
    def __init__(self, point: np.ndarray, direction: np.ndarray, name: str = "line"):
        # Create custom coordinate system where first axis is along the line
        direction = direction / np.linalg.norm(direction)
        
        # Build orthonormal basis
        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, np.array([1., 0., 0.]))
        else:
            perp1 = np.cross(direction, np.array([0., 1., 0.]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        
        basis = np.array([direction, perp1, perp2])
        coord_sys = CartesianSystem(origin=point, basis=basis, name=name)
        super().__init__(coord_sys)
    
    def contains_point(self, point: 'Point') -> bool:
        coords = point.in_system(self.coord_system)
        return abs(coords[1]) < 1e-6 and abs(coords[2]) < 1e-6
    
    def project_to_surface(self, point: 'Point') -> 'Point':
        coords = point.in_system(self.coord_system)
        return Point(np.array([coords[0], 0., 0.]), self.coord_system, surface=self)
    
    def normal_at(self, point: 'Point') -> np.ndarray:
        # Choose arbitrary perpendicular direction
        return self.coord_system.basis[1, :]
    
    def surface_distance(self, p1: 'Point', p2: 'Point') -> float:
        c1 = p1.in_system(self.coord_system)
        c2 = p2.in_system(self.coord_system)
        return abs(c2[0] - c1[0])

class Sphere(Surface):
    """Sphere with spherical coordinates centered on it"""
    
    def __init__(self, center: np.ndarray, radius: float, name: str = "sphere"):
        self.radius = radius
        coord_sys = SphericalSystem(origin=center, name=name)
        super().__init__(coord_sys)
    
    def contains_point(self, point: 'Point') -> bool:
        coords = point.in_system(self.coord_system)
        return abs(coords[0] - self.radius) < 1e-6
    
    def project_to_surface(self, point: 'Point') -> 'Point':
        cart = point.cartesian()
        center = self.coord_system.origin
        direction = cart - center
        dist = np.linalg.norm(direction)
        if dist < 1e-10:
            direction = np.array([0., 0., 1.])
            dist = 1.0
        projected_cart = center + self.radius * direction / dist
        coords = self.coord_system.from_cartesian(projected_cart)
        return Point(coords, self.coord_system, surface=self)
    
    def normal_at(self, point: 'Point') -> np.ndarray:
        cart = point.cartesian()
        normal = cart - self.coord_system.origin
        return normal / np.linalg.norm(normal)
    
    def surface_distance(self, p1: 'Point', p2: 'Point') -> float:
        """Great circle distance on sphere"""
        c1 = p1.in_system(self.coord_system)
        c2 = p2.in_system(self.coord_system)
        # Both should have r = radius, so just compute angle
        theta1, phi1 = c1[1], c1[2]
        theta2, phi2 = c2[1], c2[2]
        
        x1 = np.array([np.sin(theta1) * np.cos(phi1),
                      np.sin(theta1) * np.sin(phi1), np.cos(theta1)])
        x2 = np.array([np.sin(theta2) * np.cos(phi2),
                      np.sin(theta2) * np.sin(phi2), np.cos(theta2)])
        
        cos_angle = np.clip(np.dot(x1, x2), -1, 1)
        angle = np.arccos(cos_angle)
        return self.radius * angle

class Cylinder(Surface):
    """Cylinder with cylindrical coordinates"""
    
    def __init__(self, base_center: np.ndarray, axis: np.ndarray, 
                 radius: float, name: str = "cylinder"):
        self.radius = radius
        coord_sys = CylindricalSystem(origin=base_center, axis=axis, name=name)
        super().__init__(coord_sys)
    
    def contains_point(self, point: 'Point') -> bool:
        coords = point.in_system(self.coord_system)
        return abs(coords[0] - self.radius) < 1e-6
    
    def project_to_surface(self, point: 'Point') -> 'Point':
        coords = point.in_system(self.coord_system)
        # Keep theta and z, set r to radius
        return Point(np.array([self.radius, coords[1], coords[2]]), 
                    self.coord_system, surface=self)
    
    def normal_at(self, point: 'Point') -> np.ndarray:
        cart = point.cartesian()
        axis_origin = self.coord_system.origin
        axis_dir = self.coord_system.axis
        
        # Project point onto axis
        to_point = cart - axis_origin
        along_axis = np.dot(to_point, axis_dir) * axis_dir
        radial = to_point - along_axis
        return radial / np.linalg.norm(radial)
    
    def surface_distance(self, p1: 'Point', p2: 'Point') -> float:
        """Geodesic distance on cylinder surface"""
        c1 = p1.in_system(self.coord_system)
        c2 = p2.in_system(self.coord_system)
        
        # Both should have r = radius
        theta1, z1 = c1[1], c1[2]
        theta2, z2 = c2[1], c2[2]
        
        dtheta = np.arctan2(np.sin(theta2 - theta1), np.cos(theta2 - theta1))
        arc_length = self.radius * abs(dtheta)
        dz = abs(z2 - z1)
        
        return np.sqrt(arc_length**2 + dz**2)

class Circle(Surface):
    """Circle (1D curve embedded in 3D)"""
    
    def __init__(self, base_center: np.ndarray, axis: np.ndarray, 
                 radius: float, name: str = "circle"):
        self.radius = radius
        coord_sys = CylindricalSystem(origin=base_center, axis=axis, name=name)
        super().__init__(coord_sys)
    
    def contains_point(self, point: 'Point') -> bool:
        coords = point.in_system(self.coord_system)
        return abs(coords[0] - self.radius) < 1e-6 and coords[-1] == self.coord_system.origin[-1]
    
    def project_to_surface(self, point: 'Point') -> 'Point':
        coords = point.in_system(self.coord_system)
        # Project to plane (z=0) then to circle
        x, y = coords[0], coords[1]
        r = np.sqrt(x**2 + y**2)
        if r < 1e-10:
            x, y = self.radius, 0.
        else:
            x = x * self.radius / r
            y = y * self.radius / r
        return Point(np.array([x, y, 0.]), self.coord_system, surface=self)
    
    def normal_at(self, point: 'Point') -> np.ndarray:
        return self.normal
    
    def surface_distance(self, p1: 'Point', p2: 'Point') -> float:
        """Arc length along circle"""
        c1 = p1.in_system(self.coord_system)
        c2 = p2.in_system(self.coord_system)

        theta1 = c1[1]
        theta2 = c2[1]
        
        dtheta = np.arctan2(np.sin(theta2 - theta1), np.cos(theta2 - theta1))
        return self.radius * abs(dtheta)
    
# ============================================================================
# POINT CLASS WITH SURFACE AWARENESS
# ============================================================================

class Point:
    """Lightweight point with optional surface attachment"""
    __slots__ = ['coords', 'system_id', 'surface']
    
    def __init__(self, coords: np.ndarray, system: CoordinateSystem, 
                 surface: Optional[Surface] = None):
        self.coords = np.asarray(coords)
        self.system_id = system.id
        self.surface = surface
    
    @property
    def system(self) -> CoordinateSystem:
        return CoordinateSystemRegistry.get(self.system_id)
    
    def in_system(self, target_system: CoordinateSystem) -> np.ndarray:
        """Get coordinates in a different system"""
        if self.system_id == target_system.id:
            return self.coords
        cartesian = self.system.to_cartesian(self.coords)
        return target_system.from_cartesian(cartesian)
    
    def to_system(self, target_system: CoordinateSystem) -> 'Point':
        """Convert to another coordinate system"""
        target_coords = self.in_system(target_system)
        return Point(target_coords, target_system, surface=self.surface)
    
    def cartesian(self) -> np.ndarray:
        """Get Cartesian coordinates"""
        return self.system.to_cartesian(self.coords)
    
    def distance_to(self, other: 'Point', mode: str = 'surface') -> float:
        """Compute distance to another point
        
        Args:
            other: Target point
            mode: 'surface' = geodesic on shared surface
                  'intrinsic' = geodesic in coordinate system
                  'euclidean' = straight line in Cartesian space
        """
        if mode == 'surface':
            if self.surface is not None and self.surface is other.surface:
                return self.surface.surface_distance(self, other)
            else:
                mode = 'intrinsic'  # Fall back
        
        if mode == 'intrinsic':
            if self.system_id == other.system_id:
                return self.system.geodesic_distance(self.coords, other.coords)
            else:
                other_coords = other.in_system(self.system)
                return self.system.geodesic_distance(self.coords, other_coords)
        
        # mode == 'euclidean'
        return np.linalg.norm(other.cartesian() - self.cartesian())

class PointCloud:
    """Efficient storage for many points, optionally on a surface"""
    
    def __init__(self, coords: np.ndarray, system: CoordinateSystem,
                 surface: Optional[Surface] = None):
        self.coords = np.asarray(coords)
        self.system_id = system.id
        self.surface = surface
    
    @property
    def system(self) -> CoordinateSystem:
        return CoordinateSystemRegistry.get(self.system_id)
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(self, i) -> Point:
          return Point(self.coords[i], self.system, self.surface)
    
    def in_system(self, target_system: CoordinateSystem) -> np.ndarray:
        """Get coordinates in a different system without creating new PointCloud"""
        if self.system_id == target_system.id:
            return self.coords
        cartesian = self.system.to_cartesian(self.coords)
        return target_system.from_cartesian(cartesian)
    
    def to_system(self, target_system: CoordinateSystem) -> 'PointCloud':
        """Convert all points to another coordinate system"""
        if self.system_id == target_system.id:
            return PointCloud(self.coords.copy(), target_system, surface=self.surface)
        
        target_coords = self.in_system(target_system)
        return PointCloud(target_coords, target_system, surface=self.surface)
    
    def cartesian(self) -> np.ndarray:
        return self.system.to_cartesian(self.coords)
    
    def distances_to(self, other: 'PointCloud', mode: str = 'surface') -> np.ndarray:
        """Compute distances between point clouds"""
        if mode == 'surface' and self.surface is not None and self.surface is other.surface:
            # Compute surface distances (pairwise)
            n, m = len(self), len(other)
            if n == m:
                return np.array([
                    self.surface.surface_distance(
                        Point(self.coords[i], self.system, self.surface),
                        Point(other.coords[i], other.system, self.surface)
                    ) for i in range(n)
                ])
            else:
                distances = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        p1 = Point(self.coords[i], self.system, self.surface)
                        p2 = Point(other.coords[j], other.system, self.surface)
                        distances[i, j] = self.surface.surface_distance(p1, p2)
                return distances
        
        elif mode == 'intrinsic':
            # Use coordinate system geodesic
            if self.system_id != other.system_id:
                other_coords = other.system.to_cartesian(other.coords)
                other_coords = self.system.from_cartesian(other_coords)
            else:
                other_coords = other.coords
            
            n, m = len(self), len(other_coords)
            if n == m:
                return np.array([
                    self.system.geodesic_distance(self.coords[i], other_coords[i])
                    for i in range(n)
                ])
            else:
                distances = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        distances[i, j] = self.system.geodesic_distance(
                            self.coords[i], other_coords[j]
                        )
                return distances
        
        else:  # euclidean
            self_cart = self.cartesian()
            other_cart = other.cartesian()
            
            if len(self) == len(other):
                return np.linalg.norm(other_cart - self_cart, axis=1)
            else:
                diff = self_cart[:, np.newaxis, :] - other_cart[np.newaxis, :, :]
                return np.linalg.norm(diff, axis=2)
