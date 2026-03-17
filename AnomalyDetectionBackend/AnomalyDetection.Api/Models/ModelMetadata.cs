using System.Text.Json.Serialization;

namespace AnomalyDetection.Api.Model
{
    public class ModelMetadata
    {
        [JsonPropertyName("model_name")]
        public string ModelName { get; set; } = string.Empty;

        [JsonPropertyName("category")]
        public string Category { get; set; } = string.Empty;

        [JsonPropertyName("threshold")]
        public float Threshold { get; set; }

        [JsonPropertyName("calibration_score")]
        public float CalibrationScore { get; set; }
    }
}
