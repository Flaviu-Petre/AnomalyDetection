using AnomalyDetection.Api.Models;
using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Rewrite;

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
        public IActionResult DetectAnomaly([FromForm] string category, IFormFile image, [FromForm] bool returnHeatmap = false)
        {
            if(string.IsNullOrWhiteSpace(category))
                return BadRequest("You must provide a category (e.g., 'bottle').");

            if (image == null || image.Length == 0)
                return BadRequest("No image file was uploaded.");

            try
            {
                string normalizedCategory = category.ToLower().Trim();

                bool applyMask = !_textureClasses.Contains(normalizedCategory);

                var (mlService, threshold) = _modelManager.GetModelForCategory(category);

                using var stream = image.OpenReadStream();

                var result = mlService.PredictAnomalyScore(stream, threshold, applyMask, returnHeatmap);

                if (returnHeatmap)
                {
                    return Ok(result);
                }

                var liteResult = new AnomalyDao
                {
                    IsAnomaly = result.IsAnomaly,
                    Score = result.Score,
                    UsedThreshold = result.UsedThreshold
                };

                _statisticsService.SaveInferenceResult(
                    normalizedCategory,
                    liteResult.IsAnomaly,
                    liteResult.Score,
                    liteResult.UsedThreshold
                );

                return Ok(liteResult);
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
