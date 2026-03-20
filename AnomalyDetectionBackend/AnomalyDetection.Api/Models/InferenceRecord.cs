namespace AnomalyDetection.Api.Models
{
    public class InferenceRecord
    {
        public int Id { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public string Category { get; set; } = string.Empty;
        public bool IsAnomaly { get; set; }       
        public float Score { get; set; }     
        public float ThresholdUsed { get; set; } 
    }
}
