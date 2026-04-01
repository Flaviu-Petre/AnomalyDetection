using System.Text.Json.Serialization;

namespace AnomalyDetection.Api.Models.DTOs
{
    public class AnomalyResponse
    {
        public bool IsAnomaly { get; set; }
        public float Score { get; set; }
        public float UsedThreshold { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? HeatmapBase64 { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? PredictedCategory { get; set; }
    }
}
