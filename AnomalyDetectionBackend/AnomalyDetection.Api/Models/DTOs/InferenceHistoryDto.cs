namespace AnomalyDetection.Api.Models.DTOs
{
    public class InferenceHistoryDto
    {
        public int Id { get; set; }
        public DateTime Timestamp { get; set; }
        public string Category { get; set; } = string.Empty;
        public bool IsAnomaly { get; set; }
        public float Score { get; set; }
        public float ThresholdUsed { get; set; }
        public int UserId { get; set; }
        public string Username { get; set; } = string.Empty;
        public string ImageName { get; set; } = string.Empty;
    }
}
