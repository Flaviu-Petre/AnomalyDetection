using System.Text.Json.Serialization;

namespace AnomalyDetection.Api.Models.Domain
{
    public class ModelMetadata
    {
        [JsonPropertyName("model_name")]
        public string ModelName { get; set; } = string.Empty;

        [JsonPropertyName("category")]
        public string Category { get; set; } = string.Empty;

        [JsonPropertyName("threshold")]
        public float Threshold { get; set; }

        [JsonPropertyName("input_size")]
        public int[] InputSize { get; set; } = Array.Empty<int>();

        [JsonPropertyName("calibration_score")]
        public float CalibrationScore { get; set; }
    }
}
