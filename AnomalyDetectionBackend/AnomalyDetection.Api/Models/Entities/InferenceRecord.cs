namespace AnomalyDetection.Api.Models.Entities
{
    public class InferenceRecord
    {
        public int Id { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public string Category { get; set; } = string.Empty;
        public bool IsAnomaly { get; set; }       
        public float Score { get; set; }     
        public float ThresholdUsed { get; set; }
        public int UserId { get; set; }
    }
}
