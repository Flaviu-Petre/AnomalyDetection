namespace AnomalyDetection.Api.Models.DTOs
{
    public class DashboardStatsResponse
    {
        public int TotalInferencesThisWeek { get; set; }
        public int TotalAnomaliesThisWeek { get; set; }
        public double OverallAnomalyRatePercentage { get; set; }
        public Dictionary<string, int> AnomaliesByCategory { get; set; } = new();
        public Dictionary<string, int> InferencesByDay { get; set; } = new();
    }
}
