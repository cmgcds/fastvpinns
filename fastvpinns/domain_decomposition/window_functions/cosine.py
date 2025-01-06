from .base import WindowFunction

class CosineWindowFunction(WindowFunction):
    """
    Cosine window function.
    """
    
    def __init__(self, subdomain_mean_list, subdomain_span_list, scaling_factor, overlap_factor):
        """
        Initialize the cosine window function.
        
        Parameters
        ----------
        subdomain_mean_list : list
            List of the means of the subdomains. First dimension is the x_coordinate, second dimension is the y_coordinate.
        subdomain_span_list : list
            List of the spans of the subdomains. First dimension is the x_coordinate, second dimension is the y_coordinate.
        scaling_factor : float
            Scaling factor for the subdomains.
        overlap_factor : float
            Overlap factor for the subdomains.
        """
        super().__init__(subdomain_mean_list, subdomain_span_list, 'cosine')
        
    def get_kernel(self, x, y):
        """
        Evaluate the cosine kernel function at given coordinates.
        
        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        
        Returns
        -------
        float
            Value of the cosine kernel function at (x, y).
        """
        return np.cos(np.pi * (x - self.x_min) / self.x_span) * np.cos(np.pi * (y - self.y_min) / self.y_span)