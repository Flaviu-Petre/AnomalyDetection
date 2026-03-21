using AnomalyDetection.Api.Models;
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
        #endregion

        #region Constructor
        public FeedbackController(FeedbackService feedbackService)
        {
            _feedbackService = feedbackService;
        }
        #endregion

        #region Endpoints
        [HttpPost]
        public async Task<IActionResult> SubmitFeedback([FromForm] FeedbackRequest request)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(request.Category))
                    return BadRequest("Category is required.");

                if (request.Image == null || request.Image.Length == 0)
                    return BadRequest("Image file is required.");

                string savedPath = await _feedbackService.SaveFeedbackImageAsync(
                    request.Category,
                    request.IsActuallyAnomaly,
                    request.Image
                );

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
