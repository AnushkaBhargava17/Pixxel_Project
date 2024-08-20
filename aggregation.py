import numpy as np
from scipy.interpolate import griddata
from typing import List, Tuple

class CloudCoverAggregator:

    def spatial_aggregation(self, samples: List[Tuple[float, float, float]]) -> float:
        """Perform spatial aggregation using area-weighted averaging."""
        lats, lons, values = zip(*samples)
        
        grid_lat, grid_lon = np.mgrid[min(lats):max(lats):100j, min(lons):max(lons):100j]
        grid_values = griddata((lats, lons), values, (grid_lat, grid_lon), method='linear')
        lat_weights = np.cos(np.radians(grid_lat))
        weights = lat_weights / lat_weights.sum()
        return np.average(grid_values, weights=weights)

    def temporal_aggregation(self, time_series: List[Tuple[str, float]]) -> float:
        """Perform temporal aggregation with time-weighted averaging."""
        times, values = zip(*time_series)
        
        # Convert times to numerical values (assume ISO format strings)
        numeric_times = [time.mktime(time.strptime(t, "%Y-%m-%dT%H:%M:%S")) for t in times]
        
        time_diffs = np.diff(numeric_times)
        
        # Trapezoidal rule for integration
        integral = np.sum(time_diffs * (np.array(values[1:]) + np.array(values[:-1])) / 2)
        
        return integral / (numeric_times[-1] - numeric_times[0])

    def aggregate_cloud_cover(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float, 
                              start_time: str, end_time: str) -> float:
        """Main function to aggregate cloud cover."""
        samples = self.adaptive_grid_sampling(min_lat, max_lat, min_lon, max_lon, start_time, end_time)
        optimized_values = self.optimize_api_requests(samples, start_time, end_time)
        
        # Spatial aggregation
        spatial_aggregate = self.spatial_aggregation(list(zip(samples[:2], optimized_values)))
        
        # Temporal aggregation (for demonstration only)
        time_series = [("2024-08-17T00:00:00", spatial_aggregate),
                      ("2024-08-17T06:00:00", spatial_aggregate * 0.8),
                      ("2024-08-17T12:00:00", spatial_aggregate * 0.9),
                      ("2024-08-17T18:00:00", spatial_aggregate * 0.7)]
        temporal_aggregate = self.temporal_aggregation(time_series)
        
        return temporal_aggregate