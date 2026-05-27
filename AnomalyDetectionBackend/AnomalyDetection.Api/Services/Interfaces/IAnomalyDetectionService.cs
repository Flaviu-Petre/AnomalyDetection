using AnomalyDetection.Api.Models.Domain;

namespace AnomalyDetection.Api.Services.Interfaces
{
    public interface IAnomalyDetectionService : IDisposable
    {
        AnomalyResult PredictAnomalyScore(
            Stream imageStream,
            float threshold,
            float scoreMin,
            float scoreMax,
            bool applyMask,
            bool heatmapUseGlobalMax,
            bool returnHeatmap = false);
    }
}