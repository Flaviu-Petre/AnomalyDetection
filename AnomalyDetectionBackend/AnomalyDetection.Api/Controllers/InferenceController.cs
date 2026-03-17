using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    public class InferenceController : ControllerBase
    {
        private readonly ModelManagerService _modelManager;

        public InferenceController(ModelManagerService modelManager)
        {
            _modelManager = modelManager ?? throw new ArgumentNullException(nameof(modelManager));
        }

        [HttpPost("detect_anomaly")]
        public IActionResult DetectAnomaly([FromForm] string category, [FromForm] bool applyMask, IFormFile image)
        {
            if(string.IsNullOrWhiteSpace(category))
                return BadRequest("You must provide a category (e.g., 'bottle').");

            if (image == null || image.Length == 0)
                return BadRequest("No image file was uploaded.");

            try
            {
                var (mlService, threshold) = _modelManager.GetModelForCategory(category);

                using var stream = image.OpenReadStream();
                var result = mlService.PredictAnomalyScore(stream, threshold, applyMask);

                return Ok(result);
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
    }
}
