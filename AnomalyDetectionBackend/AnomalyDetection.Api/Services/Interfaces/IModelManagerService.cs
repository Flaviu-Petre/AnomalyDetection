using AnomalyDetection.Api.Models.Domain;
using AnomalyDetection.Api.Models.DTOs;

namespace AnomalyDetection.Api.Services.Interfaces
{
    public interface IModelManagerService
    {
        (IAnomalyDetectionService Service, ModelMetadata Metadata) GetModelForCategory(string category);
        List<ModelInfo> GetAvailableModels();
        Task UploadNewModelAsync(string category, IFormFile bankFile, IFormFile jsonMetadata);
        void DeleteModel(string category);
    }
}