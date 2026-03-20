using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    public class ModelsController : ControllerBase
    {
        private readonly ModelManagerService _modelManager;

        public ModelsController(ModelManagerService modelManager)
        {
            _modelManager = modelManager;
        }

        [HttpGet]
        public IActionResult GetModels()
        {
            var models = _modelManager.GetAvailableModels();

            return Ok(models);
        }
    }
}
