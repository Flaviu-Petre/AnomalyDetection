using System.Collections.Concurrent;
using System.Text.Json;
using AnomalyDetection.Api.Models;

namespace AnomalyDetection.Api.Services
{
    public class ModelManagerService
    {
        private readonly ConcurrentDictionary<string, AnomalyDetectionService> _activeServices = new();
        private readonly ConcurrentDictionary<string, ModelMetadata> _metadataCache = new();

        private readonly string _modelStorageDirectory = "ModelWeights";

        public (AnomalyDetectionService Service, float Threshold) GetModelForCategory(string category)
        {
            category = category.ToLower();

            if (_activeServices.ContainsKey(category))
            {
                return (_activeServices[category], _metadataCache[category].Threshold);
            }

            string modelPath = Path.Combine(_modelStorageDirectory, category, $"padim_model_{category}.onnx");
            string metaPath = Path.Combine(_modelStorageDirectory, category, $"metadata_{category}.json");

            if (!File.Exists(modelPath) || !File.Exists(metaPath))
            {
                throw new FileNotFoundException($"Could not find model or metadata files for category: '{category}'. Looked in: {modelPath}");
            }

            string json = File.ReadAllText(metaPath);
            var metadata = JsonSerializer.Deserialize<ModelMetadata>(json)
                ?? throw new Exception("Failed to parse metadata.json");

            var newService = new AnomalyDetectionService(modelPath);

            _activeServices.TryAdd(category, newService);
            _metadataCache.TryAdd(category, metadata);

            return (newService, metadata.Threshold);
        }
    }
}
