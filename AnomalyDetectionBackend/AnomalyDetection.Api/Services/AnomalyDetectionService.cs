using AnomalyDetection.Api.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Size = SixLabors.ImageSharp.Size;

namespace AnomalyDetection.Api.Services
{
    public class AnomalyDetectionService : IDisposable
    {
        private readonly InferenceSession _padimSession;

        public AnomalyDetectionService(string padimModelPath)
        {
            var options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            _padimSession = new InferenceSession(padimModelPath, options);
        }

        public AnomalyResult PredictAnomalyScore(Stream imageStream, float threshold, bool applyMask = false)
        {
            using var image = SixLabors.ImageSharp.Image.Load<Rgba32>(imageStream);

            image.Mutate(x => x
                .Resize(new ResizeOptions
                {
                    Size = new Size(256, 256),
                    Mode = ResizeMode.Stretch,
                    Sampler = KnownResamplers.Triangle
                })
                .Crop(new Rectangle(16, 16, 224, 224))
            );

            var inputTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var std = new[] { 0.229f, 0.224f, 0.225f };

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgba32> pixelRow = accessor.GetRowSpan(y);
                    for (int x = 0; x < pixelRow.Length; x++)
                    {
                        ref Rgba32 pixel = ref pixelRow[x];
                        inputTensor[0, 0, y, x] = ((pixel.R / 255f) - mean[0]) / std[0];
                        inputTensor[0, 1, y, x] = ((pixel.G / 255f) - mean[1]) / std[1];
                        inputTensor[0, 2, y, x] = ((pixel.B / 255f) - mean[2]) / std[2];
                    }
                }
            });

            // Execute Inference
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
            using var results = _padimSession.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();
            float[] rawMap = outputTensor.ToArray();

            // Smoothing
            using Mat rawMat = new Mat(224, 224, MatType.CV_32FC1);
            rawMat.SetArray(rawMap);

            using Mat blurredMat = new Mat();
            Cv2.GaussianBlur(rawMat, blurredMat, new OpenCvSharp.Size(31, 31), 8.0);

            float[] blurredMap = new float[224 * 224];
            blurredMat.GetArray(out blurredMap);

            // Contrast Masking
            if (applyMask)
            {
                float[] contrastMask = GenerateContrastMask(image);
                for (int i = 0; i < blurredMap.Length; i++)
                {
                    if (contrastMask[i] == 0f) blurredMap[i] = 0f;
                }
            }

            // 99.5th Percentile Robust Scoring
            int borderOffset = 8;
            var validScores = new List<float>();

            for (int y = borderOffset; y < 224 - borderOffset; y++)
            {
                for (int x = borderOffset; x < 224 - borderOffset; x++)
                {
                    float val = blurredMap[y * 224 + x];
                    if (val > 0) validScores.Add(val);
                }
            }

            float finalRobustScore = 0f;
            if (validScores.Count > 0)
            {
                validScores.Sort();
                int index = (int)(validScores.Count * 0.995);
                finalRobustScore = validScores[Math.Min(index, validScores.Count - 1)];
            }

            // Generating the Visual Heatmap
            string base64Heatmap = GenerateHeatmapBase64(blurredMap, image);

            return new AnomalyResult
            {
                IsAnomaly = finalRobustScore > threshold,
                Score = finalRobustScore,
                UsedThreshold = threshold,
                HeatmapBase64 = base64Heatmap
            };
        }

        // TODO: Implement the heatmap generation logic to convert the blurred anomaly map into a visual representation and encode it as a Base64 string.
        private string GenerateHeatmapBase64(float[] blurredMap, Image<Rgba32> image)
        {
            throw new NotImplementedException();
        }

        // TODO: Implement the contrast masking logic to create a binary mask based on the original image's contrast, which can be used to suppress low-contrast areas in the anomaly map.
        private float[] GenerateContrastMask(Image<Rgba32> image)
        {
            throw new NotImplementedException();
        }

        public void Dispose()
        {
            _padimSession.Dispose();
        }
    }
}
