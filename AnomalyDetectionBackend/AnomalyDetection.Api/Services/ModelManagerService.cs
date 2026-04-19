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
        private readonly ILogger<AnomalyDetectionService> _anomalyLogger;
        #endregion

        #region Constructor
        public ModelManagerService(ILogger<ModelManagerService> logger, ILogger<AnomalyDetectionService> anomalyLogger) 
        {
            _logger = logger;
            _anomalyLogger = anomalyLogger;
        }
        #endregion

        #region Public Methods
        public (AnomalyDetectionService Service, ModelMetadata Metadata) GetModelForCategory(string category)
        {
            category = category.ToLower();

            if (_activeServices.TryGetValue(category, out var svc) && _metadataCache.TryGetValue(category, out var meta))
            {
                return (svc, meta);
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

            var newService = new AnomalyDetectionService(modelPath, _anomalyLogger);

            _activeServices.TryAdd(category, newService);
            _metadataCache.TryAdd(category, metadata);

            return (newService, metadata);
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

        public async Task UploadNewModelAsync(string category, IFormFile onnxModel, IFormFile? onnxData, IFormFile jsonMetadata)
        {
            string normalizedCategory = category.ToLower().Trim();

            if (normalizedCategory.Contains("..") || normalizedCategory.Contains("/") || normalizedCategory.Contains("\\"))
            {
                throw new ArgumentException("Security Policy: Invalid category name. Path traversal characters are not allowed.");
            }

            string categoryPath = Path.Combine(_modelStorageDirectory, normalizedCategory);
            if (!Directory.Exists(categoryPath))
            {
                Directory.CreateDirectory(categoryPath);
            }

            string modelFilePath = Path.Combine(categoryPath, $"padim_model_{normalizedCategory}.onnx");
            string metaFilePath = Path.Combine(categoryPath, $"metadata_{normalizedCategory}.json");
            string dataFilePath = $"{modelFilePath}.data";

            _activeServices.TryRemove(normalizedCategory, out var oldService);
            oldService?.Dispose();
            _metadataCache.TryRemove(normalizedCategory, out _);

            if (File.Exists(dataFilePath))
            {
                File.Delete(dataFilePath);
            }

            using (var stream = new FileStream(modelFilePath, FileMode.Create))
            {
                await onnxModel.CopyToAsync(stream);
            }

            if (onnxData != null)
            {
                using (var stream = new FileStream(dataFilePath, FileMode.Create))
                {
                    await onnxData.CopyToAsync(stream);
                }
            }

            using (var stream = new FileStream(metaFilePath, FileMode.Create))
            {
                await jsonMetadata.CopyToAsync(stream);
            }

            _logger.LogInformation("Successfully uploaded and refreshed model for category: {Category}", normalizedCategory);
        }

        public void DeleteModel(string category)
        {
            string normalizedCategory = category.ToLower().Trim();

            if (normalizedCategory.Contains("..") || normalizedCategory.Contains("/") || normalizedCategory.Contains("\\"))
            {
                throw new ArgumentException("Invalid category name.");
            }

            string categoryPath = Path.Combine(_modelStorageDirectory, normalizedCategory);

            _activeServices.TryRemove(normalizedCategory, out var activeService);
            if (activeService != null)
            {
                activeService.Dispose();
            }

            _metadataCache.TryRemove(normalizedCategory, out _);

            if (Directory.Exists(categoryPath))
            {
                Directory.Delete(categoryPath, true);
                _logger.LogInformation("Deleted model directory and cleared memory for category: {Category}", normalizedCategory);
            }
            else
            {
                throw new DirectoryNotFoundException($"Model directory for '{normalizedCategory}' does not exist.");
            }
        }
        #endregion
    }
}
