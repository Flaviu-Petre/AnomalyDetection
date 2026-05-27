using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    public class FeedbackController : ControllerBase
    {
        #region Fields
        private readonly FeedbackService _feedbackService;
        private readonly ILogger<FeedbackController> _logger;
        #endregion

        #region Constructor
        public FeedbackController(FeedbackService feedbackService, ILogger<FeedbackController> logger)
        {
            _feedbackService = feedbackService;
            _logger = logger;
        }
        #endregion

        #region Endpoints

        [HttpPost]
        [Authorize]
        public async Task<IActionResult> SubmitFeedback([FromForm] FeedbackRequest request)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(request.Category))
                {
                    _logger.LogWarning("[FEEDBACK] Submission failed: Category was missing.");
                    return BadRequest("Category is required.");
                }

                if (request.Image == null || request.Image.Length == 0)
                {
                    _logger.LogWarning("[FEEDBACK] Submission failed for category '{Category}': Image file was missing.", request.Category);
                    return BadRequest("Image file is required.");
                }

                string savedPath = await _feedbackService.SaveFeedbackImageAsync(
                    request.Category,
                    request.IsActuallyAnomaly,
                    request.Image
                );

                _logger.LogInformation("[FEEDBACK] Saved feedback for category '{Category}'. Marked as Anomaly: {IsAnomaly}",
                    request.Category, request.IsActuallyAnomaly);

                return Ok(new
                {
                    Message = "Feedback saved successfully. Thank you for helping improve the model!",
                    SavedPath = savedPath
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] Failed to save feedback for category '{Category}'.", request.Category);
                return StatusCode(500, "An unexpected internal server error occurred while saving feedback.");
            }
        }

        [HttpGet("summary")]
        [Authorize(Roles = "Admin")]
        public IActionResult GetFeedbackSummary()
        {
            try
            {
                var summary = _feedbackService.GetFeedbackSummary();
                return Ok(summary);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] Failed to retrieve feedback summary.");
                return StatusCode(500, "An unexpected internal server error occurred while retrieving the feedback summary.");
            }
        }

        [HttpGet("images/{category}/{label}")]
        [Authorize(Roles = "Admin")]
        public IActionResult GetFeedbackImageList(string category, string label)
        {
            if (label != "anomaly" && label != "good")
                return BadRequest("Label must be 'anomaly' or 'good'.");

            try
            {
                var files = _feedbackService.GetFeedbackImageNames(category, label);
                return Ok(files);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] Failed to list feedback images for '{Category}/{Label}'.", category, label);
                return StatusCode(500, "An unexpected internal server error occurred.");
            }
        }

        [HttpGet("images/{category}/{label}/{filename}")]
        [Authorize(Roles = "Admin")]
        public IActionResult GetFeedbackImage(string category, string label, string filename)
        {
            if (label != "anomaly" && label != "good")
                return BadRequest("Label must be 'anomaly' or 'good'.");

            if (filename.Contains("..") || filename.Contains("/") || filename.Contains("\\"))
            {
                _logger.LogWarning("[SECURITY] Path traversal attempt blocked for filename: {Filename}", filename);
                return BadRequest("Invalid filename.");
            }

            try
            {
                var (stream, contentType) = _feedbackService.GetFeedbackImageStream(category, label, filename);
                return File(stream, contentType);
            }
            catch (FileNotFoundException)
            {
                return NotFound($"Image '{filename}' not found.");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] Failed to serve feedback image '{Filename}'.", filename);
                return StatusCode(500, "An unexpected internal server error occurred.");
            }
        }

        #endregion
    }
}