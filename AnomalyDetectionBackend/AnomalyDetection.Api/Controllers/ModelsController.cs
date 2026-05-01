using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    [Authorize]
    public class ModelsController : ControllerBase
    {
        #region Fields
        private readonly ModelManagerService _modelManager;
        private readonly ILogger<ModelsController> _logger;
        #endregion

        #region Constructor
        public ModelsController(ModelManagerService modelManager, ILogger<ModelsController> logger)
        {
            _modelManager = modelManager;
            _logger = logger;
        }
        #endregion

        #region Endpoints
        [HttpGet("get_all_models")]
        public IActionResult GetModels()
        {
            var models = _modelManager.GetAvailableModels();
            return Ok(models);
        }

        [HttpPost("upload_model")]
        [Authorize(Roles = "Admin")]
        [DisableRequestSizeLimit]
        [RequestFormLimits(MultipartBodyLengthLimit = long.MaxValue, ValueLengthLimit = int.MaxValue)]
        public async Task<IActionResult> UploadModel([FromForm] UploadModelRequest request)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(request.Category))
                {
                    _logger.LogWarning("[MODELS] Upload failed: Category is missing.");
                    return BadRequest("Category is required.");
                }

                if (request.BankFile == null || !request.BankFile.FileName.EndsWith(".npz"))
                {
                    _logger.LogWarning("[MODELS] Upload failed for category '{Category}': Invalid bank file.", request.Category);
                    return BadRequest("A valid .npz memory bank file is required.");
                }

                if (request.JsonMetadata == null || !request.JsonMetadata.FileName.EndsWith(".json"))
                {
                    _logger.LogWarning("[MODELS] Upload failed for category '{Category}': Invalid JSON file.", request.Category);
                    return BadRequest("A valid .json metadata file is required.");
                }

                await _modelManager.UploadNewModelAsync(request.Category, request.BankFile, request.JsonMetadata);

                _logger.LogInformation("[MODELS] Successfully uploaded model for category: '{Category}'", request.Category);

                return Ok(new { Message = $"Successfully uploaded and registered the new model for: {request.Category}" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] Failed to upload model for category '{Category}'.", request.Category);
                return StatusCode(500, new { Error = "An unexpected internal server error occurred while saving the model." });
            }
        }

        [HttpDelete("delete_category")]
        [Authorize(Roles = "Admin")]
        public IActionResult DeleteModel(string category)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(category))
                {
                    _logger.LogWarning("[MODELS] Delete failed: Category is missing.");
                    return BadRequest("Category is required.");
                }

                _modelManager.DeleteModel(category);

                _logger.LogInformation("[MODELS] Successfully deleted model for category: '{Category}'", category);

                return Ok(new { Message = $"Successfully deleted the model for category: {category}" });
            }
            catch (FileNotFoundException ex)
            {
                _logger.LogWarning(ex, "[MODELS] Delete failed: Files not found for category '{Category}'.", category);
                return NotFound(new { Error = "The requested model category could not be found." });
            }
            catch (ArgumentException ex)
            {
                _logger.LogWarning(ex, "[MODELS] Delete failed: Invalid argument for category '{Category}'.", category);
                return BadRequest(new { Error = "Invalid category name provided." });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] Failed to delete model for category '{Category}'.", category);
                return StatusCode(500, new { Error = "An unexpected internal server error occurred while deleting the model." });
            }
        }
        #endregion
    }
}