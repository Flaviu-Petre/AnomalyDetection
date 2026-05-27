namespace AnomalyDetection.Api.Services.Interfaces
{
    public interface IRouterService : IDisposable
    {
        Task<(string Category, float Confidence)> ClassifyAsync(Stream imageStream);
    }
}