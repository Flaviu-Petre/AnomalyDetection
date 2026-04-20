using AnomalyDetection.Api.Extensions;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    public class InferenceController : ControllerBase
    {
        #region Fields
        private readonly InferenceService _inferenceService;
        private readonly ILogger<InferenceController> _logger;
        #endregion

        #region Constructor
        public InferenceController(InferenceService inferenceService, ILogger<InferenceController> logger)
        {
            _inferenceService = inferenceService;
            _logger = logger;
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
                string imageName = image.FileName ?? "Unknown Image";

                int userId = User.GetUserId();

                var response = await _inferenceService.ProcessImageAsync(stream, imageName, userId, returnHeatmap);

                return Ok(response);
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogWarning("[INFERENCE REJECTED] Business rule validation failed: {Message}", ex.Message);
                return BadRequest(ex.Message);
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
