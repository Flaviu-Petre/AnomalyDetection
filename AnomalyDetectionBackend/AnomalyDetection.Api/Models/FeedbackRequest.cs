namespace AnomalyDetection.Api.Models
{
    public class FeedbackRequest
    {
        public string Category { get; set; } = string.Empty;
        public bool IsActuallyAnomaly { get; set; }
        public IFormFile? Image { get; set; }
    }
}
