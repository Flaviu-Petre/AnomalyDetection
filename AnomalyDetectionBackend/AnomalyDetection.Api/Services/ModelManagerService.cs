using AnomalyDetection.Api.Models.Domain;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services.Interfaces;
using System.Collections.Concurrent;
using System.Text.Json;

namespace AnomalyDetection.Api.Services
{
    public class ModelManagerService : IModelManagerService
    {
        #region Constants
        private const string EncoderPath = "PatchCore/Encoder/patchcore_model.onnx";
        private const string BanksDir = "PatchCore/Banks";
        private const string MetadataDir = "PatchCore/Metadata";
        #endregion

        #region Fields
        private readonly ConcurrentDictionary<string, IAnomalyDetectionService> _activeServices = new();
        private readonly ConcurrentDictionary<string, ModelMetadata> _metadataCache = new();
        private readonly ILogger<ModelManagerService> _logger;
        private readonly ILogger<AnomalyDetectionService> _anomalyLogger;
        #endregion

        #region Constructor
        public ModelManagerService(ILogger<ModelManagerService> logger, ILogger<AnomalyDetectionService> anomalyLogger)
        {
            _logger = logger;
            _anomalyLogger = anomalyLogger;

            if (!File.Exists(EncoderPath))
                throw new FileNotFoundException($"DINOv2 encoder not found at: {EncoderPath}");
        }
        #endregion

        #region Public Methods
        public (IAnomalyDetectionService Service, ModelMetadata Metadata) GetModelForCategory(string category)
        {
            category = category.ToLower();

            if (_activeServices.TryGetValue(category, out var svc) &&
                _metadataCache.TryGetValue(category, out var meta))
                return (svc, meta);

            string bankPath = Path.Combine(BanksDir, $"patchcore_memory_{category}.npz");
            string metaPath = Path.Combine(MetadataDir, $"metadata_{category}.json");

            if (!File.Exists(bankPath))
                throw new FileNotFoundException($"Memory bank not found for category '{category}' at: {bankPath}");
            if (!File.Exists(metaPath))
                throw new FileNotFoundException($"Metadata not found for category '{category}' at: {metaPath}");

            var metadata = JsonSerializer.Deserialize<ModelMetadata>(File.ReadAllText(metaPath))
                ?? throw new InvalidOperationException($"Failed to parse metadata for category: {category}");

            var service = new AnomalyDetectionService(
                EncoderPath,
                bankPath,
                metadata.KNeighbours,
                _anomalyLogger);

            _activeServices.TryAdd(category, service);
            _metadataCache.TryAdd(category, metadata);

            _logger.LogInformation("[MODEL MANAGER] Loaded PatchCore model for category: {Category} " +
                "(bank size: {BankSize}, k: {K})",
                category, metadata.MemoryBankSize, metadata.KNeighbours);

            return (service, metadata);
        }

        public List<ModelInfo> GetAvailableModels()
        {
            var availableModels = new List<ModelInfo>();

            if (!Directory.Exists(MetadataDir))
                return availableModels;

            foreach (var file in Directory.GetFiles(MetadataDir, "metadata_*.json"))
            {
                try
                {
                    var metadata = JsonSerializer.Deserialize<ModelMetadata>(File.ReadAllText(file));
                    if (metadata != null)
                    {
                        availableModels.Add(new ModelInfo
                        {
                            Category = metadata.ClassName,
                            Threshold = metadata.OptimalThreshold
                        });
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to parse metadata file: {File}", file);
                }
            }

            return availableModels;
        }

        public async Task UploadNewModelAsync(string category, IFormFile bankFile, IFormFile jsonMetadata)
        {
            string normalizedCategory = ValidateCategory(category);

            string bankPath = Path.Combine(BanksDir, $"patchcore_memory_{normalizedCategory}.npz");
            string metaPath = Path.Combine(MetadataDir, $"metadata_{normalizedCategory}.json");

            // Evict cached instances
            _activeServices.TryRemove(normalizedCategory, out var oldService);
            oldService?.Dispose();
            _metadataCache.TryRemove(normalizedCategory, out _);

            Directory.CreateDirectory(BanksDir);
            Directory.CreateDirectory(MetadataDir);

            using (var stream = new FileStream(bankPath, FileMode.Create))
                await bankFile.CopyToAsync(stream);

            using (var stream = new FileStream(metaPath, FileMode.Create))
                await jsonMetadata.CopyToAsync(stream);

            _logger.LogInformation("[MODEL MANAGER] Uploaded new memory bank for category: {Category}",
                normalizedCategory);
        }

        public void DeleteModel(string category)
        {
            string normalizedCategory = ValidateCategory(category);

            string bankPath = Path.Combine(BanksDir, $"patchcore_memory_{normalizedCategory}.npz");
            string metaPath = Path.Combine(MetadataDir, $"metadata_{normalizedCategory}.json");

            _activeServices.TryRemove(normalizedCategory, out var activeService);
            activeService?.Dispose();
            _metadataCache.TryRemove(normalizedCategory, out _);

            if (File.Exists(bankPath)) File.Delete(bankPath);
            if (File.Exists(metaPath)) File.Delete(metaPath);

            _logger.LogInformation("[MODEL MANAGER] Deleted model for category: {Category}", normalizedCategory);
        }
        #endregion

        #region Private Helpers
        private static string ValidateCategory(string category)
        {
            string normalized = category.ToLower().Trim();
            if (normalized.Contains("..") || normalized.Contains("/") || normalized.Contains("\\"))
                throw new ArgumentException("Invalid category name — path traversal not allowed.");
            return normalized;
        }
        #endregion
    }
}