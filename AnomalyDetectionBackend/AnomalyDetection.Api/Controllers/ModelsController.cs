using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
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
        [HttpGet]
        public IActionResult GetModels()
        {
            var models = _modelManager.GetAvailableModels();

            return Ok(models);
        }
        #endregion
    }
}
