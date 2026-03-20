namespace AnomalyDetection.Api.Models
{
    public class AnomalyDao
    {
        public bool IsAnomaly { get; set; }
        public float Score { get; set; }
        public float UsedThreshold { get; set; }
    }
}
