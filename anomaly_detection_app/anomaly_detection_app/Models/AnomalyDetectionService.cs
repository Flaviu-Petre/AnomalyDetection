using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;

public class AnomalyDetectionService : IDisposable
{
    private readonly InferenceSession _session;

    public AnomalyDetectionService(string modelPath)
    {
        // Initialize the ONNX session
        _session = new InferenceSession(modelPath);
    }

    public float PredictAnomalyScore(string imagePath)
    {
        // 1. Load Image
        using var image = Image.Load<Rgb24>(imagePath);

        // 2. Preprocess: Resize to 256x256, Center Crop to 224x224
        image.Mutate(x => x
        .Resize(new ResizeOptions
        {
            Size = new Size(256, 256),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Triangle 
        })
        .Crop(new Rectangle(16, 16, 224, 224)) 
        );


        // 3. Convert to Tensor and Normalize
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
        var mean = new[] { 0.485f, 0.456f, 0.406f };
        var std = new[] { 0.229f, 0.224f, 0.225f };

        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                var pixel = image[x, y];
                // Convert to 0-1 range and apply normalization
                inputTensor[0, 0, y, x] = ((pixel.R / 255f) - mean[0]) / std[0];
                inputTensor[0, 1, y, x] = ((pixel.G / 255f) - mean[1]) / std[1];
                inputTensor[0, 2, y, x] = ((pixel.B / 255f) - mean[2]) / std[2];
            }
        }

        // 4. Run Inference
        // The python script defines the input name as "input"
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // 5. Calculate Score: Max value of the anomaly map
        float maxScore = float.MinValue;
        foreach (var value in outputTensor)
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