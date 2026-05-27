using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services.Interfaces;

namespace AnomalyDetection.Api.Services
{
    public class InferenceService : IInferenceService
    {
        #region Fields
        private readonly IModelManagerService _modelManager;
        private readonly IStatisticsService _statisticsService;
        private readonly IRouterService _routerService;
        private readonly ILogger<InferenceService> _logger;
        #endregion

        #region Constructor
        public InferenceService(IModelManagerService modelManager, IStatisticsService statisticsService, IRouterService routerService, ILogger<InferenceService> logger)
        {
            _modelManager = modelManager;
            _statisticsService = statisticsService;
            _routerService = routerService;
            _logger = logger;
        }
        #endregion

        #region Methods
        public async Task<AnomalyResponse> ProcessImageAsync(Stream imageStream, string imageName, int userId, bool returnHeatmap)
        {
            var (normalizedCategory, confidence) = await _routerService.ClassifyAsync(imageStream);

            _logger.LogInformation("[AI ROUTER] Predicted: {Category} with {Confidence}% confidence",
                normalizedCategory, confidence * 100);

            if (normalizedCategory == "unknown")
            {
                _logger.LogWarning("[AI ROUTER REJECTED] Image not recognized. Category: {Category}, Confidence: {Confidence}%",
                    normalizedCategory, confidence * 100);
                throw new InvalidOperationException(
                    $"Image not recognized. Please upload a valid factory part. (AI Confidence was only {confidence * 100:F1}%)");
            }

            var (mlService, metadata) = _modelManager.GetModelForCategory(normalizedCategory);

            imageStream.Position = 0;

            var result = mlService.PredictAnomalyScore(
                imageStream,
                metadata.OptimalThreshold,
                metadata.ScoreMin,
                metadata.ScoreMax,
                metadata.ApplyMask,
                metadata.HeatmapUseGlobalMax,
                returnHeatmap);

            _statisticsService.SaveInferenceResult(normalizedCategory, result.IsAnomaly, result.Score, result.UsedThreshold, userId, imageName);

            _logger.LogInformation("[INFERENCE SUCCESS] Category: {Category} | Anomaly Detected: {IsAnomaly} | Score: {Score}",
                normalizedCategory, result.IsAnomaly, result.Score);

            return new AnomalyResponse
            {
                PredictedCategory = normalizedCategory,
                IsAnomaly = result.IsAnomaly,
                Score = result.Score,
                UsedThreshold = result.UsedThreshold,
                HeatmapBase64 = result.HeatmapBase64
            };
        }
        #endregion
    }
}