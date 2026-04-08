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

        private readonly string[] _textureClasses =
        [
            "carpet", "grid", "leather", "tile", "wood",
            "cable", "screw", "transistor", "zipper", "bottle"
        ];
        #endregion

        #region Constructor
        public InferenceController(ModelManagerService modelManager, StatisticsService statisticsService)
        {
            _modelManager = modelManager ?? throw new ArgumentNullException(nameof(modelManager));
            _statisticsService = statisticsService ?? throw new ArgumentNullException(nameof(statisticsService));
        }
        #endregion

        #region Endpoints
        [HttpPost("detect_anomaly")]
        [Authorize]
        public async Task<IActionResult> DetectAnomaly(IFormFile image, [FromForm] bool returnHeatmap = false)
        {
            if (image == null || image.Length == 0)
                return BadRequest("No image file was uploaded.");

            try
            {
                using var stream = image.OpenReadStream();

                var classificationResult = await AnomalyDetectionService.ClassifyImageCategoryAsync(stream);
                string normalizedCategory = classificationResult.Category.ToLower().Trim();
                float confidence = classificationResult.Confidence;

                if (normalizedCategory == "unknown" || confidence < 0.98f)
                {
                    return BadRequest("Image not recognized. Please upload a valid factory part.");
                }

                bool applyMask = !_textureClasses.Contains(normalizedCategory);

                var (mlService, threshold) = _modelManager.GetModelForCategory(normalizedCategory);

                var result = mlService.PredictAnomalyScore(stream, threshold, applyMask, returnHeatmap);

                string username = User.FindFirstValue(ClaimTypes.NameIdentifier) ?? "Unknown";

                _statisticsService.SaveInferenceResult(
                    normalizedCategory,
                    result.IsAnomaly,
                    result.Score,
                    result.UsedThreshold,
                    username
                );

                var response = new AnomalyResponse
                {
                    PredictedCategory = normalizedCategory,
                    IsAnomaly = result.IsAnomaly,
                    Score = result.Score,
                    UsedThreshold = result.UsedThreshold,
                    HeatmapBase64 = result.HeatmapBase64
                };

                return Ok(response);
            }
            catch (FileNotFoundException ex)
            {
                return NotFound(ex.Message);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"An error occurred during inference: {ex.Message}");
            }
        }
        #endregion
    }
}
