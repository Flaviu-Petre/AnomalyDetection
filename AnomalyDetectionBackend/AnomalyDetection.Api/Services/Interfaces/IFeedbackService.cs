namespace AnomalyDetection.Api.Services.Interfaces
{
    public interface IFeedbackService
    {
        Task<string> SaveFeedbackImageAsync(string category, bool isActuallyAnomaly, IFormFile image);
        List<object> GetFeedbackSummary();
        List<string> GetFeedbackImageNames(string category, string label);
        (Stream stream, string contentType) GetFeedbackImageStream(string category, string label, string filename);
    }
}