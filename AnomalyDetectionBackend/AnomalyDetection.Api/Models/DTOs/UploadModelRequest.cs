namespace AnomalyDetection.Api.Models.DTOs
{
    public class UploadModelRequest
    {
        public string Category { get; set; } = string.Empty;
        public IFormFile? OnnxModel { get; set; }
        public IFormFile? OnnxData { get; set; }
        public IFormFile? JsonMetadata { get; set; }
    }
}
