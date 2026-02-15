using anomaly_detection_app.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class AnomalyDetectionService : IDisposable
{
    private readonly InferenceSession _session;

    public AnomalyDetectionService(string modelPath)
    {
        // Initialize the ONNX session
        _session = new InferenceSession(modelPath);
    }

    public AnomalyResult PredictAnomalyScore(string imagePath)
    {
        // 1. Load Image
        using var image = Image.Load<Rgba32>(imagePath);

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
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // 5. Apply Gaussian Blur to find max score
        var (maxScore, smoothedMap) = GetMaxBlurredScore(outputTensor, 224, 224, 4f);

        // 6. Generate the Heatmap Image
        float minScore = float.MaxValue;
        foreach (var val in smoothedMap) if (val < minScore) minScore = val;

        using var heatmap = new Image<Rgba32>(224, 224);
        for (int y = 0; y < 224; y++)
        {
            for (int x = 0; x < 224; x++)
            {
                float v = smoothedMap[y, x];

                float normalized = (v - minScore) / (maxScore - minScore + 1e-5f);

                float r = Math.Clamp(1.5f - Math.Abs(4.0f * normalized - 3.0f), 0f, 1f);
                float g = Math.Clamp(1.5f - Math.Abs(4.0f * normalized - 2.0f), 0f, 1f);
                float b = Math.Clamp(1.5f - Math.Abs(4.0f * normalized - 1.0f), 0f, 1f);

                heatmap[x, y] = new Rgba32((byte)(r * 255), (byte)(g * 255), (byte)(b * 255), 150);
            }
        }

        image.Mutate(ctx => ctx.DrawImage(heatmap, PixelColorBlendingMode.Normal, PixelAlphaCompositionMode.SrcOver, 1.0f));

        using var ms = new MemoryStream();
        image.SaveAsJpeg(ms);

        return new AnomalyResult
        {
            Score = maxScore,
            HeatmapImageBytes = ms.ToArray()
        };
    }

    private (float MaxScore, float[,] SmoothedMap) GetMaxBlurredScore(Tensor<float> outputTensor, int width = 224, int height = 224, float sigma = 4f)
    {
        int radius = (int)Math.Ceiling(4 * sigma);
        int size = 2 * radius + 1;
        float[] kernel = new float[size];
        float sum = 0;

        for (int i = 0; i < size; i++)
        {
            int x = i - radius;
            kernel[i] = (float)Math.Exp(-(x * x) / (2 * sigma * sigma));
            sum += kernel[i];
        }
        for (int i = 0; i < size; i++) kernel[i] /= sum;

        float[,] map = new float[height, width];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                map[y, x] = outputTensor[0, 0, y, x];

        float[,] temp = new float[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float val = 0;
                for (int i = 0; i < size; i++)
                {
                    int px = Math.Clamp(x + i - radius, 0, width - 1);
                    val += map[y, px] * kernel[i];
                }
                temp[y, x] = val;
            }
        }

        float[,] finalMap = new float[height, width];
        float maxScore = float.MinValue;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float val = 0;
                for (int i = 0; i < size; i++)
                {
                    int py = Math.Clamp(y + i - radius, 0, height - 1);
                    val += temp[py, x] * kernel[i];
                }
                finalMap[y, x] = val;
                if (val > maxScore) maxScore = val;
            }
        }

        return (maxScore, finalMap);
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}