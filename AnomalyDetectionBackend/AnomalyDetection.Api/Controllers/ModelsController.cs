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
        #endregion

        #region Constructor
        public ModelsController(ModelManagerService modelManager)
        {
            _modelManager = modelManager;
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
                    return BadRequest("Category is required.");

                if (request.OnnxModel == null || !request.OnnxModel.FileName.EndsWith(".onnx"))
                    return BadRequest("A valid .onnx model file is required.");

                if (request.JsonMetadata == null || !request.JsonMetadata.FileName.EndsWith(".json"))
                    return BadRequest("A valid .json metadata file is required.");

                await _modelManager.UploadNewModelAsync(request.Category, request.OnnxModel, request.OnnxData, request.JsonMetadata);

                return Ok(new { Message = $"Successfully uploaded and registered the new model for: {request.Category}" });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal server error while saving model: {ex.Message}");
            }
        }

        [HttpDelete("delete_category")]
        [Authorize(Roles = "Admin")]
        public IActionResult DeleteModel(string category)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(category))
                    return BadRequest("Category is required.");

                _modelManager.DeleteModel(category);

                return Ok(new { Message = $"Successfully deleted the model and cleared memory for category: {category}" });
            }
            catch (DirectoryNotFoundException ex)
            { 
                return NotFound(new { Error = ex.Message });
            }
            catch (ArgumentException ex)
            {
                return BadRequest(new { Error = ex.Message });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { Error = $"Internal server error while deleting model: {ex.Message}" });
            }
        }

        #endregion
    }
}
