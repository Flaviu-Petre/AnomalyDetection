using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Rewrite;
using System.Security.Claims;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    public class InferenceController : ControllerBase
    {
        #region Fields
        private readonly ModelManagerService _modelManager;
        private readonly StatisticsService _statisticsService;
        private readonly ILogger<InferenceController> _logger;

        private readonly string[] _textureClasses =
        [
            "carpet", "grid", "leather", "tile", "wood",
            "cable", "screw", "transistor", "zipper", "bottle"
        ];
        #endregion

        #region Constructor
        public InferenceController(ModelManagerService modelManager, StatisticsService statisticsService, ILogger<InferenceController> logger)
        {
            _modelManager = modelManager ?? throw new ArgumentNullException(nameof(modelManager));
            _statisticsService = statisticsService ?? throw new ArgumentNullException(nameof(statisticsService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        #endregion

        #region Endpoints
        [HttpPost("detect_anomaly")]
        [Authorize]
        public async Task<IActionResult> DetectAnomaly(IFormFile image, [FromForm] bool returnHeatmap = false)
        {
            if (image == null || image.Length == 0)
            {
                _logger.LogWarning("[INFERENCE] Request rejected: No image file was uploaded.");
                return BadRequest("No image file was uploaded.");
            }

            try
            {
                using var stream = image.OpenReadStream();

                var classificationResult = await AnomalyDetectionService.ClassifyImageCategoryAsync(stream);
                string normalizedCategory = classificationResult.Category.ToLower().Trim();
                float confidence = classificationResult.Confidence;

                _logger.LogInformation("[AI ROUTER] Predicted: {Category} with {Confidence}% confidence", normalizedCategory, confidence * 100);

                if (normalizedCategory == "unknown" || confidence < 0.93f)
                {
                    _logger.LogWarning("[AI ROUTER REJECTED] Image failed threshold. Category: {Category}, Confidence: {Confidence}%", normalizedCategory, confidence * 100);
                    return BadRequest($"Image not recognized. Please upload a valid factory part. (AI Confidence was only {confidence * 100:F1}%)");
                }

                bool applyMask = !_textureClasses.Contains(normalizedCategory);

                var (mlService, threshold) = _modelManager.GetModelForCategory(normalizedCategory);

                var result = mlService.PredictAnomalyScore(stream, threshold, applyMask, returnHeatmap);

                int userId = 0;
                var userIdStr = User.FindFirstValue("id");
                if (!string.IsNullOrEmpty(userIdStr))
                {
                    int.TryParse(userIdStr, out userId);
                }

                string imageName = image.FileName ?? "Unknown Image";

                _statisticsService.SaveInferenceResult(
                    normalizedCategory,
                    result.IsAnomaly,
                    result.Score,
                    result.UsedThreshold,
                    userId,
                    imageName
                );

                var response = new AnomalyResponse
                {
                    PredictedCategory = normalizedCategory,
                    IsAnomaly = result.IsAnomaly,
                    Score = result.Score,
                    UsedThreshold = result.UsedThreshold,
                    HeatmapBase64 = result.HeatmapBase64
                };

                _logger.LogInformation("[INFERENCE SUCCESS] Category: {Category} | Anomaly Detected: {IsAnomaly} | Score: {Score}",
                    normalizedCategory, result.IsAnomaly, result.Score);

                return Ok(response);
            }
            catch (FileNotFoundException ex)
            {
                _logger.LogWarning(ex, "[INFERENCE ERROR] Model files not found for category.");
                return NotFound("The AI model for this category is currently unavailable.");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] An unexpected error occurred during Padim inference.");
                return StatusCode(500, "An unexpected internal server error occurred while processing the image.");
            }
        }
        #endregion
    }
}
