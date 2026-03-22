namespace AnomalyDetection.Api.Models.Domain
{
    public class AnomalyResult
    {
        public bool IsAnomaly { get; set; }
        public float Score { get; set; }
        public float UsedThreshold { get; set; }
        public string? HeatmapBase64 { get; set; }
    }
}
