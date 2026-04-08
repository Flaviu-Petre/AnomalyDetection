using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services;
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

                _logger.LogInformation("[FEEDBACK] Successfully saved user feedback for category '{Category}'. Marked as Anomaly: {IsAnomaly}",
                    request.Category,
                    request.IsActuallyAnomaly);

                return Ok(new
                {
                    Message = "Feedback saved successfully. Thank you for helping improve the model!",
                    SavedPath = savedPath
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal server error while saving feedback: {ex.Message}");
            }
        }
        #endregion
    }
}
