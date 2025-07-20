"""
Mathematics Plugin for AETHERION
Provides advanced mathematical operations, statistical analysis, and computational capabilities
"""

import math
import statistics
from typing import Dict, List, Any, Union, Optional
import logging
import random
from collections import defaultdict

from .plugin_base import PluginBase, PluginMetadata, PluginConfig

logger = logging.getLogger(__name__)


class MathPlugin(PluginBase):
    """Advanced mathematical computation plugin"""
    
    name = "math_processor"
    version = "1.0.0"
    description = "Advanced mathematical operations and statistical analysis"
    
    metadata = PluginMetadata(
        name=name,
        version=version,
        description=description,
        author="AETHERION Core",
        category="computation",
        tags=["math", "statistics", "analysis", "computation"],
        dependencies=[],
        config_schema={
            "precision": {"type": "integer", "default": 6},
            "enable_advanced": {"type": "boolean", "default": True},
            "max_iterations": {"type": "integer", "default": 1000},
            "tolerance": {"type": "float", "default": 1e-10}
        }
    )
    
    def __init__(self):
        super().__init__()
        self.config = PluginConfig(self.metadata.config_schema)
        self._initialized = False
        
    def initialize(self):
        """Initialize the math plugin"""
        logger.info("Initializing Math Plugin")
        self._initialized = True
        
    def round_to_precision(self, value: float) -> float:
        """Round value to configured precision"""
        precision = self.config.get("precision", 6)
        return round(value, precision)
    
    def basic_operations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic mathematical operations"""
        a = data.get("a", 0)
        b = data.get("b", 0)
        operation = data.get("operation", "add")
        
        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else float('inf'),
            "power": a ** b,
            "modulo": a % b if b != 0 else float('nan'),
            "floor_divide": a // b if b != 0 else float('inf')
        }
        
        result = operations.get(operation, 0)
        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": self.round_to_precision(result)
        }
    
    def statistical_analysis(self, data: List[float]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        if not data:
            return {"error": "No data provided for analysis"}
            
        try:
            # Basic statistics
            mean = statistics.mean(data)
            median = statistics.median(data)
            mode = statistics.mode(data) if len(set(data)) < len(data) else None
            
            # Variance and standard deviation
            variance = statistics.variance(data)
            stdev = statistics.stdev(data)
            
            # Min, max, range
            min_val = min(data)
            max_val = max(data)
            data_range = max_val - min_val
            
            # Percentiles
            sorted_data = sorted(data)
            q1 = sorted_data[len(sorted_data) // 4]
            q3 = sorted_data[3 * len(sorted_data) // 4]
            iqr = q3 - q1
            
            # Skewness and kurtosis (simplified)
            n = len(data)
            skewness = sum(((x - mean) / stdev) ** 3 for x in data) / n if stdev > 0 else 0
            kurtosis = sum(((x - mean) / stdev) ** 4 for x in data) / n if stdev > 0 else 0
            
            return {
                "count": len(data),
                "mean": self.round_to_precision(mean),
                "median": self.round_to_precision(median),
                "mode": mode,
                "variance": self.round_to_precision(variance),
                "standard_deviation": self.round_to_precision(stdev),
                "minimum": self.round_to_precision(min_val),
                "maximum": self.round_to_precision(max_val),
                "range": self.round_to_precision(data_range),
                "q1": self.round_to_precision(q1),
                "q3": self.round_to_precision(q3),
                "iqr": self.round_to_precision(iqr),
                "skewness": self.round_to_precision(skewness),
                "kurtosis": self.round_to_precision(kurtosis)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return {"error": str(e)}
    
    def advanced_math(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced mathematical operations"""
        if not self.config.get("enable_advanced", True):
            return {"error": "Advanced math operations disabled"}
            
        operation = data.get("operation", "")
        value = data.get("value", 0)
        
        try:
            if operation == "factorial":
                if value < 0:
                    return {"error": "Factorial not defined for negative numbers"}
                result = math.factorial(int(value))
                
            elif operation == "log":
                base = data.get("base", 10)
                if value <= 0:
                    return {"error": "Logarithm not defined for non-positive numbers"}
                if base == 10:
                    result = math.log10(value)
                elif base == 2:
                    result = math.log2(value)
                else:
                    result = math.log(value, base)
                    
            elif operation == "sqrt":
                if value < 0:
                    return {"error": "Square root not defined for negative numbers"}
                result = math.sqrt(value)
                
            elif operation == "sin":
                result = math.sin(value)
                
            elif operation == "cos":
                result = math.cos(value)
                
            elif operation == "tan":
                result = math.tan(value)
                
            elif operation == "asin":
                if not -1 <= value <= 1:
                    return {"error": "Arcsin domain is [-1, 1]"}
                result = math.asin(value)
                
            elif operation == "acos":
                if not -1 <= value <= 1:
                    return {"error": "Arccos domain is [-1, 1]"}
                result = math.acos(value)
                
            elif operation == "atan":
                result = math.atan(value)
                
            elif operation == "exp":
                result = math.exp(value)
                
            elif operation == "gamma":
                if value <= 0:
                    return {"error": "Gamma function not defined for non-positive numbers"}
                result = math.gamma(value)
                
            else:
                return {"error": f"Unknown operation: {operation}"}
                
            return {
                "operation": operation,
                "value": value,
                "result": self.round_to_precision(result)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced math operation {operation}: {e}")
            return {"error": str(e)}
    
    def matrix_operations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform matrix operations"""
        matrix_a = data.get("matrix_a", [])
        matrix_b = data.get("matrix_b", [])
        operation = data.get("operation", "add")
        
        if not matrix_a:
            return {"error": "Matrix A is required"}
            
        try:
            if operation == "transpose":
                if not matrix_a or not matrix_a[0]:
                    return {"error": "Invalid matrix"}
                    
                rows = len(matrix_a)
                cols = len(matrix_a[0])
                result = [[matrix_a[j][i] for j in range(rows)] for i in range(cols)]
                
            elif operation == "determinant":
                if len(matrix_a) != len(matrix_a[0]):
                    return {"error": "Determinant only defined for square matrices"}
                    
                result = self._calculate_determinant(matrix_a)
                
            elif operation == "inverse":
                if len(matrix_a) != len(matrix_a[0]):
                    return {"error": "Inverse only defined for square matrices"}
                    
                det = self._calculate_determinant(matrix_a)
                if abs(det) < self.config.get("tolerance", 1e-10):
                    return {"error": "Matrix is not invertible (determinant is zero)"}
                    
                result = self._calculate_inverse(matrix_a)
                
            elif operation == "multiply":
                if not matrix_b:
                    return {"error": "Matrix B is required for multiplication"}
                    
                result = self._multiply_matrices(matrix_a, matrix_b)
                
            else:
                return {"error": f"Unknown matrix operation: {operation}"}
                
            return {
                "operation": operation,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error in matrix operation {operation}: {e}")
            return {"error": str(e)}
    
    def _calculate_determinant(self, matrix: List[List[float]]) -> float:
        """Calculate determinant of a matrix"""
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
            
        det = 0
        for j in range(n):
            minor = [[matrix[i][k] for k in range(n) if k != j] for i in range(1, n)]
            det += matrix[0][j] * self._calculate_determinant(minor) * (-1) ** j
        return det
    
    def _calculate_inverse(self, matrix: List[List[float]]) -> List[List[float]]:
        """Calculate inverse of a matrix using adjugate method"""
        n = len(matrix)
        det = self._calculate_determinant(matrix)
        
        # Calculate adjugate matrix
        adj = []
        for i in range(n):
            adj_row = []
            for j in range(n):
                minor = [[matrix[k][l] for l in range(n) if l != j] for k in range(n) if k != i]
                cofactor = self._calculate_determinant(minor) * (-1) ** (i + j)
                adj_row.append(cofactor)
            adj.append(adj_row)
            
        # Transpose adjugate and divide by determinant
        adj_transpose = [[adj[j][i] for j in range(n)] for i in range(n)]
        inverse = [[adj_transpose[i][j] / det for j in range(n)] for i in range(n)]
        
        return inverse
    
    def _multiply_matrices(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Multiply two matrices"""
        rows_a, cols_a = len(a), len(a[0])
        rows_b, cols_b = len(b), len(b[0])
        
        if cols_a != rows_b:
            raise ValueError("Matrix dimensions incompatible for multiplication")
            
        result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
        
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]
                    
        return result
    
    def numerical_methods(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform numerical methods"""
        method = data.get("method", "")
        function_str = data.get("function", "")
        x0 = data.get("x0", 0)
        tolerance = self.config.get("tolerance", 1e-10)
        max_iter = self.config.get("max_iterations", 1000)
        
        try:
            if method == "newton":
                # Newton's method for finding roots
                def f(x):
                    return eval(function_str, {"x": x, "math": math})
                    
                def f_prime(x):
                    h = 1e-8
                    return (f(x + h) - f(x - h)) / (2 * h)
                    
                x = x0
                for i in range(max_iter):
                    fx = f(x)
                    if abs(fx) < tolerance:
                        break
                    fpx = f_prime(x)
                    if abs(fpx) < tolerance:
                        return {"error": "Derivative too close to zero"}
                    x = x - fx / fpx
                    
                return {
                    "method": "newton",
                    "root": self.round_to_precision(x),
                    "iterations": i + 1,
                    "function_value": self.round_to_precision(f(x))
                }
                
            elif method == "bisection":
                # Bisection method
                a = data.get("a", -1)
                b = data.get("b", 1)
                
                def f(x):
                    return eval(function_str, {"x": x, "math": math})
                    
                if f(a) * f(b) > 0:
                    return {"error": "Function values at endpoints must have opposite signs"}
                    
                for i in range(max_iter):
                    c = (a + b) / 2
                    fc = f(c)
                    if abs(fc) < tolerance:
                        break
                    if f(a) * fc < 0:
                        b = c
                    else:
                        a = c
                        
                return {
                    "method": "bisection",
                    "root": self.round_to_precision(c),
                    "iterations": i + 1,
                    "function_value": self.round_to_precision(fc)
                }
                
            else:
                return {"error": f"Unknown numerical method: {method}"}
                
        except Exception as e:
            logger.error(f"Error in numerical method {method}: {e}")
            return {"error": str(e)}
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mathematical operations based on input"""
        if not self._initialized:
            self.initialize()
            
        operation_type = input_data.get("type", "basic")
        
        try:
            if operation_type == "basic":
                return self.basic_operations(input_data)
            elif operation_type == "statistics":
                data = input_data.get("data", [])
                return self.statistical_analysis(data)
            elif operation_type == "advanced":
                return self.advanced_math(input_data)
            elif operation_type == "matrix":
                return self.matrix_operations(input_data)
            elif operation_type == "numerical":
                return self.numerical_methods(input_data)
            else:
                return {"error": f"Unknown operation type: {operation_type}"}
                
        except Exception as e:
            logger.error(f"Error in math plugin execution: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup plugin resources"""
        logger.info("Cleaning up Math Plugin")
        self._initialized = False 