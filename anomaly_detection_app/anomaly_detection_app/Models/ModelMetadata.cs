using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json.Serialization;

namespace anomaly_detection_app.Models
{
    public class ModelMetadata
    {
        [JsonPropertyName("model_name")]
        public string ModelName { get; set; }

        [JsonPropertyName("threshold")]
        public float Threshold { get; set; }

        [JsonPropertyName("input_size")]
        public List<int> InputSize { get; set; }

        [JsonPropertyName("category")]
        public string Category { get; set; }
    }
}
