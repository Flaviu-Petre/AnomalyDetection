using System.Collections.Concurrent;
using System.Text.Json;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Models.Domain;

namespace AnomalyDetection.Api.Services
{
    public class ModelManagerService
    {
        #region Variables
        private readonly ConcurrentDictionary<string, AnomalyDetectionService> _activeServices = new();
        private readonly ConcurrentDictionary<string, ModelMetadata> _metadataCache = new();
        private readonly string _modelStorageDirectory = "ModelWeights";
        private readonly ILogger<ModelManagerService> _logger;
        #endregion

        #region Constructor
        public ModelManagerService(ILogger<ModelManagerService> logger)
        {
            _logger = logger;
        }
        #endregion

        #region Public Methods
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

        public List<ModelInfo> GetAvailableModels()
        {
            var availableModels = new List<ModelInfo>();
            if (!Directory.Exists(_modelStorageDirectory))
            {
                return availableModels;
            }

            var categoryFolders = Directory.GetDirectories(_modelStorageDirectory);

            foreach (var folder in categoryFolders)
            {
                string category = new DirectoryInfo(folder).Name;
                string metaPath = Path.Combine(folder, $"metadata_{category}.json");

                if (File.Exists(metaPath))
                {
                    try
                    {
                        string json = File.ReadAllText(metaPath);
                        var metadata = JsonSerializer.Deserialize<ModelMetadata>(json);

                        if (metadata != null)
                        {
                            availableModels.Add(new ModelInfo
                            {
                                Category = metadata.Category,
                                Threshold = metadata.Threshold
                            });
                        }
                    }
                    catch (Exception ex) 
                    {
                        _logger.LogWarning(ex, "Failed to load or parse metadata for category '{Category}'. The file might be corrupted.", category);
                    }
                }
            }

            return availableModels;
        }
        #endregion
    }
}
