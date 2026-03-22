namespace AnomalyDetection.Api.Models.DTOs
{
    public class AnomalyResponse
    {
        public bool IsAnomaly { get; set; }
        public float Score { get; set; }
        public float UsedThreshold { get; set; }
    }
}
