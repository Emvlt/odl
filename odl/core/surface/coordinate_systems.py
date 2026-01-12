"""Coordinate System class"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

# ============================================================================
# COORDINATE SYSTEM INFRASTRUCTURE
# ============================================================================

class CoordinateSystemRegistry:
    """Singleton registry for coordinate systems"""
    _systems: Dict[int, 'CoordinateSystem'] = {}
    _next_id: int = 0
    
    @classmethod
    def register(cls, system: 'CoordinateSystem') -> int:
        system_id = cls._next_id
        cls._systems[system_id] = system
        cls._next_id += 1
        return system_id
    
    @classmethod
    def get(cls, system_id: int) -> 'CoordinateSystem':
        return cls._systems[system_id]
    
    @classmethod
    def clear(cls):
        cls._systems.clear()
        cls._next_id = 0

class CoordinateSystem(ABC):
    """Base class for coordinate systems"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.id = CoordinateSystemRegistry.register(self)
    
    @abstractmethod
    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """Convert from this coordinate system to Cartesian"""
        pass
    
    @abstractmethod
    def from_cartesian(self, cartesian: np.ndarray) -> np.ndarray:
        """Convert from Cartesian to this coordinate system"""
        pass
    
    @abstractmethod
    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        """Compute metric tensor at given coordinates"""
        pass
    
    def geodesic_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Compute intrinsic geodesic distance between two points"""
        def integrand(t):
            pos = coords1 + t * (coords2 - coords1)
            velocity = coords2 - coords1
            g = self.metric_tensor(pos)
            return np.sqrt(velocity @ g @ velocity)
        
        from scipy.integrate import quad
        distance, _ = quad(integrand, 0, 1, limit=100)
        return distance

class CartesianSystem(CoordinateSystem):
    """Standard Cartesian coordinate system"""
    
    def __init__(self, origin: np.ndarray = None, basis: np.ndarray = None, name: str = "cartesian"):
        self.origin = origin if origin is not None else np.zeros(3)
        self.basis = basis if basis is not None else np.eye(3)
        super().__init__(name)
    
    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        if coords.ndim == 1:
            return self.origin + self.basis.T @ coords
        # For array of points: (N, 3) @ (3, 3).T + (3,)
        return (coords @ self.basis.T) + self.origin[np.newaxis, :]
    
    def from_cartesian(self, cartesian: np.ndarray) -> np.ndarray:
        if cartesian.ndim == 1:
            return self.basis @ (cartesian - self.origin)
        # For array of points: (N, 3) - (3,) then @ (3, 3)
        return (cartesian - self.origin[np.newaxis, :]) @ self.basis
    
    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        return np.eye(3)
    
    def geodesic_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        return np.linalg.norm(coords2 - coords1)

class CylindricalSystem(CoordinateSystem):
    """Cylindrical coordinates (r, theta, z)"""
    
    def __init__(self, origin: np.ndarray = None, axis: np.ndarray = None, name: str = "cylindrical"):
        self.origin = origin if origin is not None else np.zeros(3)
        self.axis = axis if axis is not None else np.array([0., 0., 1.])
        self.axis = self.axis / np.linalg.norm(self.axis)
        
        if abs(self.axis[0]) < 0.9:
            radial_ref = np.cross(self.axis, np.array([1., 0., 0.]))
        else:
            radial_ref = np.cross(self.axis, np.array([0., 1., 0.]))
        self.radial_ref = radial_ref / np.linalg.norm(radial_ref)
        super().__init__(name)
    
    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        if coords.ndim == 1:
            r, theta, z = coords
            x_local = r * np.cos(theta)
            y_local = r * np.sin(theta)
            perp = np.cross(self.axis, self.radial_ref)
            pos = x_local * self.radial_ref + y_local * perp + z * self.axis
            return self.origin + pos
        else:
            r, theta, z = coords[:, 0], coords[:, 1], coords[:, 2]
            x_local = r * np.cos(theta)
            y_local = r * np.sin(theta)
            perp = np.cross(self.axis, self.radial_ref)
            pos = (x_local[:, np.newaxis] * self.radial_ref[np.newaxis, :] + 
                   y_local[:, np.newaxis] * perp[np.newaxis, :] + 
                   z[:, np.newaxis] * self.axis[np.newaxis, :])
            return self.origin[np.newaxis, :] + pos
    
    def from_cartesian(self, cartesian: np.ndarray) -> np.ndarray:
        if cartesian.ndim == 1:
            relative = cartesian - self.origin
            z = np.dot(relative, self.axis)
            radial_vec = relative - z * self.axis
            r = np.linalg.norm(radial_vec)
            if r < 1e-10:
                theta = 0.0
            else:
                perp = np.cross(self.axis, self.radial_ref)
                x_local = np.dot(radial_vec, self.radial_ref)
                y_local = np.dot(radial_vec, perp)
                theta = np.arctan2(y_local, x_local)
            return np.array([r, theta, z])
        else:
            relative = cartesian - self.origin[np.newaxis, :]
            z = relative @ self.axis
            radial_vec = relative - z[:, np.newaxis] * self.axis[np.newaxis, :]
            r = np.linalg.norm(radial_vec, axis=1)
            perp = np.cross(self.axis, self.radial_ref)
            x_local = radial_vec @ self.radial_ref
            y_local = radial_vec @ perp
            theta = np.arctan2(y_local, x_local)
            return np.column_stack([r, theta, z])
    
    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        r = coords[0] if coords.ndim == 1 else coords[:, 0]
        if np.isscalar(r):
            return np.diag([1.0, r**2, 1.0])
        return np.array([np.diag([1.0, r_i**2, 1.0]) for r_i in r])
    
    def geodesic_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        r1, theta1, z1 = coords1
        r2, theta2, z2 = coords2
        dr, dz = r2 - r1, z2 - z1
        dtheta = np.arctan2(np.sin(theta2 - theta1), np.cos(theta2 - theta1))
        
        if abs(dr) < 1e-10:
            r_avg = (r1 + r2) / 2
            arc_length = r_avg * abs(dtheta)
            return np.sqrt(arc_length**2 + dz**2)
        return super().geodesic_distance(coords1, coords2)

class SphericalSystem(CoordinateSystem):
    """Spherical coordinates (r, theta, phi)"""
    
    def __init__(self, origin: np.ndarray = None, name: str = "spherical"):
        self.origin = origin if origin is not None else np.zeros(3)
        super().__init__(name)
    
    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        if coords.ndim == 1:
            r, theta, phi = coords
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return self.origin + np.array([x, y, z])
        else:
            r, theta, phi = coords[:, 0], coords[:, 1], coords[:, 2]
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return self.origin[np.newaxis, :] + np.column_stack([x, y, z])
    
    def from_cartesian(self, cartesian: np.ndarray) -> np.ndarray:
        if cartesian.ndim == 1:
            relative = cartesian - self.origin
            r = np.linalg.norm(relative)
            if r < 1e-10:
                return np.array([0., 0., 0.])
            theta = np.arccos(np.clip(relative[2] / r, -1, 1))
            phi = np.arctan2(relative[1], relative[0])
            return np.array([r, theta, phi])
        else:
            relative = cartesian - self.origin[np.newaxis, :]
            r = np.linalg.norm(relative, axis=1)
            theta = np.arccos(np.clip(relative[:, 2] / (r + 1e-10), -1, 1))
            phi = np.arctan2(relative[:, 1], relative[:, 0])
            return np.column_stack([r, theta, phi])
    
    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        if coords.ndim == 1:
            r, theta, _ = coords
            return np.diag([1.0, r**2, (r * np.sin(theta))**2])
        r, theta = coords[:, 0], coords[:, 1]
        return np.array([np.diag([1.0, r_i**2, (r_i * np.sin(theta_i))**2]) 
                       for r_i, theta_i in zip(r, theta)])
    
    def geodesic_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        r1, theta1, phi1 = coords1
        r2, theta2, phi2 = coords2
        
        if abs(r1 - r2) < 1e-10:
            x1 = np.array([np.sin(theta1) * np.cos(phi1),
                          np.sin(theta1) * np.sin(phi1), np.cos(theta1)])
            x2 = np.array([np.sin(theta2) * np.cos(phi2),
                          np.sin(theta2) * np.sin(phi2), np.cos(theta2)])
            cos_angle = np.clip(np.dot(x1, x2), -1, 1)
            return r1 * np.arccos(cos_angle)
        return super().geodesic_distance(coords1, coords2)