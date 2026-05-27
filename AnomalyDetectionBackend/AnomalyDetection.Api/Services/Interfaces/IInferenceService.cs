using AnomalyDetection.Api.Models.DTOs;

namespace AnomalyDetection.Api.Services.Interfaces
{
    public interface IInferenceService
    {
        Task<AnomalyResponse> ProcessImageAsync(Stream imageStream, string imageName, int userId, bool returnHeatmap);
    }
}