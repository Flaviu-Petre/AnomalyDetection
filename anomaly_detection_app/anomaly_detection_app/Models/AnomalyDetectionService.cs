using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;

namespace anomaly_detection_app.Models
{
    public class AnomalyDetectionService : IDisposable
    {
        private readonly InferenceSession _session;

        public AnomalyDetectionService(string modelPath)
        {
            _session = new InferenceSession(modelPath);
        }

        public float PredictAnomalyScore(string imagePath)
        {
            using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath);

            image.Mutate(x => x
            .Resize(new ResizeOptions
            {
                Size = new SixLabors.ImageSharp.Size(224, 224),
                Mode = ResizeMode.Crop
            })
            .Crop(new SixLabors.ImageSharp.Rectangle(16, 16, 224, 224)));

            var inputTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var std = new[] { 0.229f, 0.224f, 0.225f };

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image[x, y];
                    inputTensor[0, 0, y, x] = ((pixel.R / 255f) - mean[0]) / std[0];
                    inputTensor[0, 1, y, x] = ((pixel.G / 255f) - mean[1]) / std[1];
                    inputTensor[0, 2, y, x] = ((pixel.B / 255f) - mean[2]) / std[2];
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using var results = _session.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();

            float maxScore = float.MinValue;
            foreach( var value in outputTensor)
            {
                if (value > maxScore) maxScore = value;
            }

            return maxScore;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
