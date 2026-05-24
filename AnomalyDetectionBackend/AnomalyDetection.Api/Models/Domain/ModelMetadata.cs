using System.Text.Json.Serialization;

namespace AnomalyDetection.Api.Models.Domain
{
    public class ModelMetadata
    {
        [JsonPropertyName("class_name")]
        public string ClassName { get; set; } = string.Empty;

        [JsonPropertyName("model_name")]
        public string ModelName { get; set; } = string.Empty;

        [JsonPropertyName("feature_dim")]
        public int FeatureDim { get; set; }

        [JsonPropertyName("grid_size")]
        public int GridSize { get; set; }

        [JsonPropertyName("k_neighbours")]
        public int KNeighbours { get; set; }

        [JsonPropertyName("optimal_threshold")]
        public float OptimalThreshold { get; set; }

        [JsonPropertyName("score_min")]
        public float ScoreMin { get; set; }

        [JsonPropertyName("score_max")]
        public float ScoreMax { get; set; }
        [JsonPropertyName("apply_mask")]
        public bool ApplyMask { get; set; }

        [JsonPropertyName("heatmap_use_global_max")]
        public bool HeatmapUseGlobalMax { get; set; } = true;

        [JsonPropertyName("image_auroc")]
        public float ImageAuroc { get; set; }

        [JsonPropertyName("pixel_auroc")]
        public float PixelAuroc { get; set; }

        [JsonPropertyName("memory_bank_size")]
        public int MemoryBankSize { get; set; }

        [JsonPropertyName("bank_file")]
        public string BankFile { get; set; } = string.Empty;

        [JsonPropertyName("preprocessing")]
        public PreprocessingConfig Preprocessing { get; set; } = new();
    }

    public class PreprocessingConfig
    {
        [JsonPropertyName("resize")]
        public int Resize { get; set; }

        [JsonPropertyName("crop")]
        public int Crop { get; set; }

        [JsonPropertyName("mean")]
        public float[] Mean { get; set; } = Array.Empty<float>();

        [JsonPropertyName("std")]
        public float[] Std { get; set; } = Array.Empty<float>();
    }
}